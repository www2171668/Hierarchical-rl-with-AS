from gym.envs.registration import register

# % 注册环境模拟类到gym
register(
    id='MBRLCartpole-v0',   # 注册号，调用时使用
    entry_point='dmbrl.env.cartpole:CartpoleEnv'  # dmbrl是文件夹名，cartpole是文件名，CartpoleEnv是类名
)

register(
    id='MBRLReacher3D-v0',
    entry_point='dmbrl.env.reacher:Reacher3DEnv'
)

register(
    id='MBRLPusher-v0',
    entry_point='dmbrl.env.pusher:PusherEnv'
)

register(
    id='MBRLHalfCheetah-v0',
    entry_point='dmbrl.env.half_cheetah:HalfCheetahEnv'
)

# % 调用方法说明
# import dmbrl
# env = gym.make('MBRLCartpole-v0')