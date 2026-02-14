import abc
import enum
import gymnasium.spaces as spaces
import numpy as np

import util.torch_util as torch_util

class EnvMode(enum.Enum):
    TRAIN = 0
    TEST = 1

class DoneFlags(enum.Enum):
    NULL = 0
    FAIL = 1
    SUCC = 2
    TIME = 3

class BaseEnv(abc.ABC):
    def __init__(self, visualize):
        self._mode = EnvMode.TRAIN
        self._visualize = visualize
        self._action_space = None
        self._diagnostics = dict()
        return
    
    @abc.abstractmethod
    def reset(self, env_ids=None):
        return
    
    @abc.abstractmethod
    def step(self, action):
        return
    
    def get_obs_space(self):
        obs, _ = self.reset()
        obs_shape = list(obs.shape[1:])
        obs_dtype = torch_util.torch_dtype_to_numpy(obs.dtype)
        obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=obs_dtype,
        )
        return obs_space

    def get_action_space(self):
        return self._action_space

    def set_mode(self, mode):
        self._mode = mode
        return

    def get_num_envs(self):
        return int(1)

    def get_reward_fail(self):
        return 0.0

    def get_reward_succ(self):
        return 0.0

    def get_visualize(self):
        return self._visualize
    
    def get_env_time(self, env_ids=None):
        return 0.0
    
    def record_diagnostics(self):
        return self._diagnostics