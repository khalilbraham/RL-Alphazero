from typing import Optional
from copy import deepcopy

import numpy as np

import gymnasium as gym
from gymnasium import spaces, logger

from utils import seq_is_schur


class SchurEnv(gym.Env):
    """
    Environment representing a Schur state task.

    Parameters:
    - env_config (dict): Configuration dictionary containing:
        - "n_partition" (int): Number of partitions in the Schur partition.
        - "max_per_partition" (int): Maximum elements allowed per partition.
    """
    def __init__(self, env_config):
        """
        Initializes the SchurEnv instance.

        Parameters:
        - env_config (dict): Configuration dictionary containing:
            - "n_partition" (int): Number of partitions in the Schur partition.
            - "max_per_partition" (int): Maximum elements allowed per partition.
        """
        self.n_partition = env_config["n_partition"]
        self.max_per_part = env_config["max_per_partition"]
        self.action_space = spaces.Discrete(self.n_partition)
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(low=0, high=self.n_partition, shape=(self.max_per_part,), dtype=np.int32),
                "action_mask": spaces.Box(low=0, high=1, shape=(self.n_partition,), dtype=np.int8)
            }
        )
        self.state = None
        self.max_n = 0
        self.running_reward = 0
        self.mask = None
        self.steps_beyond_terminated = None

    def step(self, action):
        """
        Executes a step in the environment.

        Parameters:
        - action (int): The action to be taken, representing the state.

        Returns:
        - Tuple: A tuple containing:
            - observation (np.ndarray): The new state of the environment.
            - reward (float): The reward obtained for the current step.
            - terminated (bool): Whether the episode is terminated.
            - info (dict): Additional information.
        """
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        self.state[self.max_n] = action
        self.max_n += 1
        
        terminated = self.max_n == self.max_per_part
        if terminated:
            action_mask = np.zeros(self.n_partition, dtype=np.int8)
        else:
            self.update_mask(action)
            action_mask = self.mask[self.max_n]
            terminated = action_mask.sum() == 0

        if terminated:
            if self.steps_beyond_terminated is None:
                self.steps_beyond_terminated = 0
                # reward = 1.0
            else:
                if self.steps_beyond_terminated == 0:
                    logger.warn(
                        "You are calling 'step()' even though this "
                        "environment has already returned terminated = True. You "
                        "should always call 'reset()' once you receive 'terminated = "
                        "True' -- any further steps are undefined behavior."
                    )
                self.steps_beyond_terminated += 1
                print(f"steps_beyond_terminated {self.steps_beyond_terminated}")
        
        if not terminated:
            self.running_reward += 1

        if self.max_n == self.max_per_part:
            self.running_reward += 20

        score = self.running_reward if terminated else 0
        return (
            {
                "obs": deepcopy(self.state),
                "action_mask": deepcopy(action_mask),
                # "max_n": self.max_n
            },
            score,
            terminated,
            False,
            {},
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """
        Resets the environment to its initial state.

        Parameters:
        - seed (int, optional): Seed for the random number generator.
        - options (dict, optional): Additional options.

        Returns:
        - Tuple: A tuple containing:
            - observation (np.ndarray): The initial state of the environment.
            - info (dict): Additional information.
        """
        super().reset(seed=seed)

        self.state = np.ones(self.max_per_part, dtype=np.int32) * self.n_partition
        self.state[0] = 0
        self.running_reward = 0
        self.mask = np.ones((self.max_per_part, self.n_partition), dtype=np.int8)
        self.max_n = 1
        self.update_mask(0)
        self.steps_beyond_terminated = None
        action_mask = self.mask[self.max_n]
        return {
            "obs": deepcopy(self.state),
            "action_mask": deepcopy(action_mask),
            }, {}

    def isTerminated(self):
        """
        Checks if the episode is terminated.

        Returns:
        - bool: True if the episode is terminated, False otherwise.
        """
        return (not seq_is_schur(deepcopy(self.state), self.n_partition)) or self.max_n == self.max_per_part

    def set_state(self, state):
        self.state = deepcopy(state[0])
        self.running_reward = state[1]
        self.max_n = state[2]
        self.mask = deepcopy(state[3])
        self.steps_beyond_terminated = state[4]
        action_mask = self.mask[self.max_n]
        return {
            "obs": deepcopy(self.state),
            "action_mask": deepcopy(action_mask),
            }

    def get_state(self):
        return deepcopy(self.state), self.running_reward, self.max_n, deepcopy(self.mask), self.steps_beyond_terminated

    def compute_action_mask(self, state, max_n):
        mask = []
        for i in range(self.n_partition):
            tmp = deepcopy(state)
            tmp[max_n] = i
            m = seq_is_schur(deepcopy(tmp), self.n_partition)
            mask.append(int(m))
        return np.array(mask, dtype=np.int8)

    def update_mask(self, color):
        mid = min(self.max_n, self.max_per_part - self.max_n)
        for idx in range(mid):
            if self.state[idx] == color:
                idx_1 = idx + self.max_n
                self.mask[idx_1][color] = 0
