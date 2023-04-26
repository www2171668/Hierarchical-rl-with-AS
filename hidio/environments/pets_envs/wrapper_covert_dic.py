import gym
import numpy as np
from gym import error, spaces

class ConvertDictWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        super(ConvertDictWrapper, self).__init__(env)

        self.observation_space = spaces.Dict(dict(
            observation=self.observation_space,
            desired_goal=spaces.Box(-np.inf, np.inf, shape=self.goal.shape, dtype='float32'),
        ))

    def observation(self, observation):
        return {'observation': observation, 'desired_goal': self.goal}

    def get_grip_ob(self):
        return self.grip_ob
