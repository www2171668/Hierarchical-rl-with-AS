import gym

class SuccessWrapper(gym.Wrapper):

    def __init__(self, env, max_episode_steps):
        super().__init__(env)
        self._max_episode_steps = max_episode_steps

    def reset(self, **kwargs):
        self._steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._steps += 1

        info["success"] = 0.0
        if self._steps >= self._max_episode_steps and info["is_success"] == 1:
            info["success"] = 1.0

        info.pop("is_success")
        return obs, reward, done, info