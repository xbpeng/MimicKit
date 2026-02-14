import engines.engine_builder as engine_builder

import abc
import enum
import gymnasium.spaces as spaces
import numpy as np
import torch

import envs.base_env as base_env
import util.torch_util as torch_util
import util.camera as camera

class PlayMode(enum.Enum):
    PLAY = 0
    ONE_STEP = 1
    PAUSE = 2

class SimEnv(base_env.BaseEnv):
    NAME = "sim_env"

    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        super().__init__(visualize=visualize)

        self._device = device
        self._episode_length = env_config["episode_length"] # episode length in seconds
        
        self._engine = self._build_engine(engine_config, num_envs, device, visualize, record_video)
        self._build_envs(env_config, num_envs)
        self._engine.initialize_sim()
        
        self._action_space = self._build_action_space()
        self._build_sim_tensors(env_config)
        self._build_data_buffers()

        if self._visualize:
            self._build_camera(env_config)
            self._play_mode = PlayMode.PLAY
            self._setup_gui()

        return
    
    def get_obs_space(self):
        obs = self._compute_obs()
        obs_shape = list(obs.shape[1:])
        obs_dtype = torch_util.torch_dtype_to_numpy(obs.dtype)
        obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=obs_dtype,
        )
        return obs_space
    
    def set_mode(self, mode):
        super().set_mode(mode)

        if (self._engine.enabled_record_video()):
            if (mode == base_env.EnvMode.TEST):
                self._engine.start_video_recording()
            else:
                self._engine.stop_video_recording()

        return

    def reset(self, env_ids=None):
        if (env_ids is None):
            reset_env_ids = self._env_ids
        else:
            reset_env_ids = env_ids

        self._reset_envs(reset_env_ids)
        
        self._update_observations(env_ids)
        self._update_info(env_ids)

        return self._obs_buf, self._info
    
    def step(self, action):
        # apply actions
        self._pre_physics_step(action)

        self._physics_step()
        
        # compute observations, rewards, resets, ...
        self._post_physics_step()

        if (self._visualize):
            self._render()
        
        return self._obs_buf, self._reward_buf, self._done_buf, self._info
    
    def get_num_envs(self):
        return self._engine.get_num_envs()
    
    def get_env_time(self, env_ids=None):
        if (env_ids is None):
            env_time = self._time_buf
        else:
            env_time = self._time_buf[env_ids]
        return env_time
    
    def record_diagnostics(self):
        if (self._engine.enabled_record_video()):
            self._diagnostics["sim_recording"] = self._engine.get_video_recording()

        diag = super().record_diagnostics()
        return diag
    
    def _pre_physics_step(self, actions):
        self._apply_action(actions)
        return
    
    def _physics_step(self):
        self._step_sim()
        return
    
    def _render(self):
        while True:
            self._update_camera()
            self._render_scene()

            if (self._play_mode != PlayMode.PAUSE):
                break
        
        if (self._play_mode == PlayMode.ONE_STEP):
            self._play_mode = PlayMode.PAUSE
        return
    
    def _render_scene(self):
        self._engine.render()
        return
    
    def _step_sim(self):
        self._engine.step()
        return
    
    def _update_misc(self):
        return

    def _update_info(self, env_ids=None):
        return
    
    def _update_camera(self):
        self._camera.update()
        return

    @abc.abstractmethod
    def _apply_action(self, actions):
        return
    
    @abc.abstractmethod
    def _update_reward(self):
        return
    
    @abc.abstractmethod
    def _update_done(self):
        return
    
    @abc.abstractmethod
    def _compute_obs(env_ids=None):
        return
    
    def _update_observations(self, env_ids=None):
        if (env_ids is None or len(env_ids) > 0):
            obs = self._compute_obs(env_ids)
            if (env_ids is None):
                self._obs_buf[:] = obs
            else:
                self._obs_buf[env_ids] = obs
        return

    def _post_physics_step(self):
        self._update_time()
        self._update_misc()
        self._update_observations()
        self._update_info()
        self._update_reward()
        self._update_done()
        return

    def _build_engine(self, engine_config, num_envs, device, visualize, record_video=False):
        engine = engine_builder.build_engine(engine_config, num_envs, device, visualize, record_video=record_video)
        return engine
    
    @abc.abstractmethod
    def _build_envs(self, config, num_envs):
        return
    
    def _build_sim_tensors(self, config):
        num_envs = self.get_num_envs()
        self._env_ids = torch.arange(num_envs, device=self._device, dtype=torch.long)
        return

    def _build_data_buffers(self):
        num_envs = self.get_num_envs()

        self._reward_buf = torch.zeros(num_envs, device=self._device, dtype=torch.float)
        self._done_buf = torch.zeros(num_envs, device=self._device, dtype=torch.int)
        self._timestep_buf = torch.zeros(num_envs, device=self._device, dtype=torch.int)
        self._time_buf = torch.zeros(num_envs, device=self._device, dtype=torch.float)

        obs_space = self.get_obs_space()
        obs_dtype = torch_util.numpy_dtype_to_torch(obs_space.dtype)
        self._obs_buf = torch.zeros([num_envs] + list(obs_space.shape), device=self._device, dtype=obs_dtype)

        self._info = dict()
        return
    
    @abc.abstractmethod
    def _reset_envs(self, env_ids):
        if (len(env_ids) > 0):
            self._timestep_buf[env_ids] = 0
            self._time_buf[env_ids] = 0
            self._done_buf[env_ids] = base_env.DoneFlags.NULL.value
        return
    
    def _update_time(self):
        self._timestep_buf += 1
        self._time_buf[:] = self._engine.get_timestep() * self._timestep_buf
        return

    @abc.abstractmethod
    def _build_action_space(self):
        return
    
    def _build_camera(self, env_config):
        cam_pos = np.array([0.0, -5.0, 3.0])
        cam_target = np.array([0.0, 0.0, 0.0])
        cam_mode = env_config.get("camera_mode", "still")
        cam_mode = camera.CameraMode[cam_mode]

        self._camera = camera.Camera(mode=cam_mode,
                                     engine=self._engine,
                                     pos=cam_pos,
                                     target=cam_target)
        return
    
    def _setup_gui(self):
        def toggle_play():
            if (self._play_mode == PlayMode.PLAY):
                self._play_mode = PlayMode.PAUSE
            else:
                self._play_mode = PlayMode.PLAY
            return
        self._engine.register_keyboard_callback("ENTER", toggle_play)

        def one_step():
            self._play_mode = PlayMode.ONE_STEP
            return
        self._engine.register_keyboard_callback("SPACE", one_step)
        return