import argparse
import yaml

from alpha_zero import AlphaZero, AlphaZeroConfig
from model import Model
import ray
from ray import air, tune
from env import SchurEnv


def train(
    env_config: dict,
    rollouts: dict,
    train_batch_size: int,
    sgd_minibatch_size: int,
    lr: float,
    num_sgd_iter: int,
    mcts_config: dict,
    ranked_rewards: dict,
    model_config: dict,
    num_gpus: int,
    timesteps_total: int
):
    ray.init()

    config = (
        AlphaZeroConfig()
        .rollouts(
            num_rollout_workers=rollouts["num_rollout_workers"],
            rollout_fragment_length=rollouts["rollout_fragment_length"],
        )
        .framework("torch")
        .environment(
            SchurEnv,
            env_config=env_config
        )
        .training(
            train_batch_size=train_batch_size,
            sgd_minibatch_size=sgd_minibatch_size,
            lr=lr,
            vf_share_layers=True,
            num_sgd_iter=num_sgd_iter,
            mcts_config=mcts_config,
            ranked_rewards=ranked_rewards,
            model={
                "custom_model": Model,
                "custom_model_config": model_config,
            }
        )
        .resources(
            num_gpus=num_gpus
        )
    )

    # stop_reward = 40.0

    tuner = tune.Tuner(
        AlphaZero,
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            stop={
                # "sampler_results/episode_reward_mean": stop_reward,
                "timesteps_total": timesteps_total,
            },
            failure_config=air.FailureConfig(fail_fast="raise"),
        ),
    )
    results = tuner.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="trainer")
    parser.add_argument(
        "--config_file",
        type=str,
        default="train_config.yaml",
        help="Path to yaml config file"
    )
    args = parser.parse_args()
    config_path = args.config_file
    with open(config_path, "r") as f:
        kwargs = yaml.safe_load(f)
    train(**kwargs)
