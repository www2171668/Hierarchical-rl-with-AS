from gym.envs.registration import register

register(
    id='Reacher-v1',
    entry_point='envs.pets_envs.reacher:Reacher3DEnv',
    max_episode_steps=100
)

register(
    id='Pusher-v1',
    entry_point='pets_envs.pusher:PusherEnv',
    max_episode_steps=100
)
