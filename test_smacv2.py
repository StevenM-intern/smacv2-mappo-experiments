from smacv2.env import StarCraft2Env

env = StarCraft2Env(
    map_name="3m",
    seed=1
)

obs, state = env.reset()
print("Reset successful.")

env.close()
