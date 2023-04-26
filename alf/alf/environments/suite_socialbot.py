try:
    import social_bot
    # The following import is to allow gin config of environments take effects
    import social_bot.envs
except ImportError:
    social_bot = None

import contextlib
from fasteners.process_lock import InterProcessLock
import functools
import gym
import socket
import numpy as np

import alf
from alf.environments import suite_gym, alf_wrappers, process_environment
from alf.environments.utils import UnwrappedEnvChecker

DEFAULT_SOCIALBOT_PORT = 11345

_unwrapped_env_checker_ = UnwrappedEnvChecker()

def is_available():
    return social_bot is not None

class SuccessWrapper(gym.Wrapper):
    """Retrieve the success info from the environment return.
    Fetch使用,本质上也是稀疏奖励.与robot_env一样记录了is_success"""

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
        # only count success at the episode end.
        # 如果 _max_episode_steps 为0,则需要在记录成功率时对info["success"]求均值
        if self._steps >= self._max_episode_steps and info["is_success"] == 1:
            info["success"] = 1.0

        info.pop("is_success")  # from gym, we remove it here
        return obs, reward, done, info

@alf.configurable
def load(environment_name,
         env_id=None,
         port=None,
         wrap_with_process=False,
         discount=1.0,
         max_episode_steps=None,
         gym_env_wrappers=(),
         alf_env_wrappers=()):
    """Loads the selected environment and wraps it with the specified wrappers.

    Note that by default a TimeLimit wrapper is used to limit episode lengths
    to the default benchmarks defined by the registered environments.

    Args:
        environment_name (str): Name for the environment to load.
        env_id (int): (optional) ID of the environment.
        port (int): Port used for the environment
        wrap_with_process (bool): Whether wrap environment in a new process
        discount (float): Discount to use for the environment.
        max_episode_steps (int): If None the max_episode_steps will be set to the default
            step limit defined in the environment's spec. No limit is applied if set
            to 0 or if there is no timestep_limit set in the environment's spec.
        gym_env_wrappers (Iterable): Iterable with references to gym_wrappers,
            classes to use directly on the gym environment.
        alf_env_wrappers (Iterable): Iterable with references to alf_wrappers
            classes to use on the ALF environment.

    Returns:
        An AlfEnvironmentBase instance.
    """
    _unwrapped_env_checker_.check_and_update(wrap_with_process)
    if gym_env_wrappers is None:
        gym_env_wrappers = ()
    if alf_env_wrappers is None:
        alf_env_wrappers = ()

    gym_spec = gym.spec(environment_name)
    if max_episode_steps is None:
        if gym_spec.max_episode_steps is not None:
            max_episode_steps = gym_spec.max_episode_steps
        else:
            max_episode_steps = 0

    def env_ctor(port, env_id=None):
        gym_env = gym_spec.make(port=port)
        gym_env = SuccessWrapper(gym_env, 0)  # 到达即停止,并记录成功情况
        return suite_gym.wrap_env(
            gym_env,
            env_id=env_id,
            discount=discount,
            max_episode_steps=max_episode_steps,
            gym_env_wrappers=gym_env_wrappers,
            alf_env_wrappers=alf_env_wrappers)

    port_range = [port, port + 1] if port else [DEFAULT_SOCIALBOT_PORT]
    with _get_unused_port(*port_range) as port:
        if wrap_with_process:
            process_env = process_environment.ProcessEnvironment(
                functools.partial(env_ctor, port))
            process_env.start()
            torch_env = alf_wrappers.AlfEnvironmentBaseWrapper(process_env)
        else:
            torch_env = env_ctor(port=port, env_id=env_id)
    return torch_env

@contextlib.contextmanager
def _get_unused_port(start, end=65536, n=1):
    """Get an unused port in the range [start, end) .

    Args:
        start (int) : port range start
        end (int): port range end
        n (int): get ``n`` consecutive unused ports
    Raises:
        socket.error: if no unused port is available
    """
    process_locks = []
    unused_ports = []
    try:
        for port in range(start, end):
            process_locks.append(
                InterProcessLock(path='/tmp/socialbot/{}.lock'.format(port)))
            if not process_locks[-1].acquire(blocking=False):
                process_locks[-1].lockfile.close()
                process_locks.pop()
                for process_lock in process_locks:
                    process_lock.release()
                process_locks = []
                continue
            try:
                with contextlib.closing(socket.socket()) as sock:
                    sock.bind(('', port))
                    unused_ports.append(port)
                    if len(unused_ports) == 2:
                        break
            except socket.error:
                for process_lock in process_locks:
                    process_lock.release()
                process_locks = []
        if len(unused_ports) < n:
            raise socket.error("No unused port in [{}, {})".format(start, end))
        if n == 1:
            yield unused_ports[0]
        else:
            yield unused_ports
    finally:
        if process_locks:
            for process_lock in process_locks:
                process_lock.release()
