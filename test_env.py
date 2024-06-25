from env import SchurEnv


env_config = {
    "n_partition": 3,
    "max_per_partition": 13
}


env = SchurEnv(env_config)
a = env.reset()
print(a)
b = env.step(1)
print(b)