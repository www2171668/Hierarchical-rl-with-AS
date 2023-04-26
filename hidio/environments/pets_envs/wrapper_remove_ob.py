import numpy as np
from gym import error, spaces

class RemoveDesWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        super(RemoveDesWrapper, self).__init__(env)

    def observation(self, observation):
        shape = observation['desired_goal'].shape
        observation['desired_goal'] = np.zeros(shape)
        return observation