from gymnasium.envs.registration import register

register(
    id="gymnasium_env/DroneTsp-v1",
    entry_point="gymnasium_env.envs:DroneTspEnv",
)
