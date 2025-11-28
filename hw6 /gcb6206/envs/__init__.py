from gymnasium.envs.registration import register

def register_envs():
    register(
        id='cheetah-gcb6206-v0',
        entry_point='gcb6206.envs.cheetah:HalfCheetahEnv',
        max_episode_steps=1000,
    )
