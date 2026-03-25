import abc
import enum
import gymnasium.spaces as spaces
import numpy as np
import os
import time
import torch

import envs.base_env as base_env
import learning.experience_buffer as experience_buffer
import learning.mp_optimizer as mp_optimizer
import learning.normalizer as normalizer
import learning.return_tracker as return_tracker
from util.logger import Logger
import util.mp_util as mp_util
import util.torch_util as torch_util
import util.logger as logger
import util.tb_logger as tb_logger
import util.wandb_logger as wandb_logger

import learning.distribution_gaussian_diag as distribution_gaussian_diag

class AgentMode(enum.Enum):
    TRAIN = 0
    TEST = 1

class BaseAgent(torch.nn.Module):
    def __init__(self, config, env, device):
        super().__init__()

        self._env = env
        self._device = device
        self._iter = 0
        self._sample_count = 0
        self._config = config
        self._load_params(config)

        self._build_normalizers()
        self._build_model(config)
        self.to(self._device)

        self._build_optimizer(config)

        self._build_exp_buffer(config)
        self._build_return_tracker()
        
        self._mode = AgentMode.TRAIN
        self._curr_obs = None
        self._curr_info = None
        return

    def train_model(self, max_samples, out_dir, save_int_models, logger_type):
        start_time = time.time()

        out_model_file = os.path.join(out_dir, "model.pt")
        log_file = os.path.join(out_dir, "log.txt")
        self._logger = self._build_logger(logger_type, log_file, self._config)
        
        if (save_int_models):
            int_out_dir = os.path.join(out_dir, "int_models")
            if (mp_util.is_root_proc() and not os.path.exists(int_out_dir)):
                os.makedirs(int_out_dir, exist_ok=True)
        else:
            int_out_dir = ""
        
        self._curr_obs, self._curr_info = self._reset_envs()
        self._init_train()

        while self._sample_count < max_samples:
            train_info = self._train_iter()
            
            self._sample_count = self._update_sample_count()
            output_iter = (self._iter % self._iters_per_output == 0) or (self._sample_count >= max_samples)

            if (output_iter):
                test_info = self.test_model(self._test_episodes)
            
            env_diag_info = self._env.record_diagnostics()
            self._log_train_info(train_info, test_info, env_diag_info, start_time) 
            self._logger.print_log()

            if (output_iter):
                self._logger.write_log()
                self._output_train_model(self._iter, out_model_file, int_out_dir)

                self._train_return_tracker.reset()
                self._curr_obs, self._curr_info = self._reset_envs()
            
            self._iter += 1

        return

    def test_model(self, num_episodes):
        self.eval()
        self.set_mode(AgentMode.TEST)
        
        num_procs = mp_util.get_num_procs()
        num_eps_proc = int(np.ceil(num_episodes / num_procs))

        with torch.no_grad():
            self._curr_obs, self._curr_info = self._reset_envs()
            test_info = self._rollout_test(num_eps_proc)

        return test_info
    
    def get_action_size(self):
        a_space = self._env.get_action_space()
        if (isinstance(a_space, spaces.Box)):
            a_size = np.prod(a_space.shape)
        elif (isinstance(a_space, spaces.Discrete)):
            a_size = 1
        else:
            assert(False), "Unsuppoted action space: {}".format(a_space)
        return a_size
    
    def set_mode(self, mode):
        self._mode = mode
        if (self._mode == AgentMode.TRAIN):
            self._env.set_mode(base_env.EnvMode.TRAIN)
        elif (self._mode == AgentMode.TEST):
            self._env.set_mode(base_env.EnvMode.TEST)
        else:
            assert(False), "Unsupported agent mode: {}".format(mode)
        return

    def get_num_envs(self):
        return self._env.get_num_envs()

    def save(self, out_file):
        if (mp_util.is_root_proc()):
            state_dict = self.state_dict()
            torch.save(state_dict, out_file)
        return

    def load(self, in_file):
        state_dict = torch.load(in_file, map_location=self._device)
        self.load_state_dict(state_dict)
        self._sync_optimizer()
        Logger.print("Loaded model parameters from {:s}".format(in_file))
        return

    def calc_num_params(self):
        params = self.parameters()
        num_params = sum(p.numel() for p in params if p.requires_grad)
        return num_params
    
    def _load_params(self, config):
        self._discount = config["discount"]
        self._iters_per_output = config["iters_per_output"]
        self._normalizer_samples = config.get("normalizer_samples", np.inf)
        self._test_episodes = config["test_episodes"]
        
        self._steps_per_iter = config["steps_per_iter"]
        return

    def _build_normalizers(self):
        obs_space = self._env.get_obs_space()
        obs_dtype = torch_util.numpy_dtype_to_torch(obs_space.dtype)
        self._obs_norm = normalizer.Normalizer(obs_space.shape, clip=10.0, device=self._device, dtype=obs_dtype)

        self._a_norm = self._build_action_normalizer()
        return
    
    def _build_action_normalizer(self):
        a_space = self._env.get_action_space()
        a_dtype = torch_util.numpy_dtype_to_torch(a_space.dtype)

        if (isinstance(a_space, spaces.Box)):
            a_mean = torch.tensor(0.5 * (a_space.high + a_space.low), device=self._device, dtype=a_dtype)
            a_std = torch.tensor(0.5 * (a_space.high - a_space.low), device=self._device, dtype=a_dtype)
            
            # ensure initialized std is strictly greater than 0 to avoid degenerate normalizer
            assert (a_std > 0).all().item(), "init_std must be > 0 for action normalizer (Box action space wrong! Check your XML file. Joints must have 'limited=true' and non-zero bounds.)"

            a_norm = normalizer.Normalizer(a_mean.shape, device=self._device, init_mean=a_mean, 
                                                 init_std=a_std, dtype=a_dtype)
        elif (isinstance(a_space, spaces.Discrete)):
            a_mean = torch.tensor([0], device=self._device, dtype=a_dtype)
            a_std = torch.tensor([1], device=self._device, dtype=a_dtype)
            a_norm = normalizer.Normalizer(a_mean.shape, device=self._device, init_mean=a_mean, 
                                                 init_std=a_std, min_std=0, dtype=a_dtype)
        else:
            assert(False), "Unsupported action space: {}".format(a_space)

        return a_norm
    
    def _build_exp_buffer(self, config):
        buffer_length = self._get_exp_buffer_length()
        batch_size = self.get_num_envs()
        self._exp_buffer = experience_buffer.ExperienceBuffer(buffer_length=buffer_length, batch_size=batch_size,
                                                              device=self._device)
        return

    def _build_return_tracker(self):
        self._train_return_tracker = return_tracker.ReturnTracker(self.get_num_envs(), self._device)
        self._test_return_tracker = return_tracker.ReturnTracker(self.get_num_envs(), self._device)
        return

    def _build_logger(self, logger_type, log_file, config):
        if (logger_type == "txt"):
            log = logger.Logger()
        elif (logger_type == "tb"):
            log = tb_logger.TBLogger()
        elif (logger_type == "wandb"):
            log = wandb_logger.WandbLogger("mimickit", config)
        else:
            assert(False), "Unsupported logger: {:s}".format(logger_type)

        log.set_step_key("Samples")
        if (mp_util.is_root_proc()):
            log.configure_output_file(log_file)
        
        return log

    def _update_sample_count(self):
        sample_count = self._exp_buffer.get_total_samples()
        sample_count = mp_util.reduce_sum(sample_count)
        return sample_count
    
    def _init_train(self):
        self._iter = 0
        self._sample_count = 0
        self._exp_buffer.clear()
        self._train_return_tracker.reset()
        self._test_return_tracker.reset()
        return

    def _train_iter(self):
        self._init_iter()
        
        self.eval()
        self.set_mode(AgentMode.TRAIN)

        with torch.no_grad():
            self._rollout_train(self._steps_per_iter)
        
        data_info = self._build_train_data()
        train_info = self._update_model()
        
        if (self._need_normalizer_update()):
            self._update_normalizers()

        info = {**train_info, **data_info}
        
        info["mean_return"] = self._train_return_tracker.get_mean_return().item()
        info["mean_ep_len"] = self._train_return_tracker.get_mean_ep_len().item()
        info["num_eps"] = self._train_return_tracker.get_episodes()
        
        return info

    def _init_iter(self):
        return

    def _rollout_train(self, num_steps):
        for i in range(num_steps):
            action, action_info = self._decide_action(self._curr_obs, self._curr_info)
            self._record_data_pre_step(self._curr_obs, self._curr_info, action, action_info)

            next_obs, r, done, next_info = self._step_env(action)
            self._train_return_tracker.update(r, done)
            self._record_data_post_step(next_obs, r, done, next_info)
            
            self._curr_obs, self._curr_info = self._reset_done_envs(done)
            self._exp_buffer.inc()
        return
    
    def _rollout_test(self, num_episodes):
        self._test_return_tracker.reset()

        if (num_episodes == 0):
            test_info = {
                "mean_return": 0.0,
                "mean_ep_len": 0.0,
                "num_eps": 0
            }
        else:
            num_envs = self.get_num_envs()
            # minimum number of episodes to collect per env
            # this is mitigate bias in the return estimate towards shorter episodes
            min_eps_per_env = int(np.ceil(num_episodes / num_envs))

            while True:
                action, action_info = self._decide_action(self._curr_obs, self._curr_info)

                next_obs, r, done, next_info = self._step_env(action)
                self._test_return_tracker.update(r, done)
            
                self._curr_obs, self._curr_info = self._reset_done_envs(done)
            
                eps_per_env = self._test_return_tracker.get_eps_per_env()
                if (torch.all(eps_per_env > min_eps_per_env - 1)):
                    break
        
            test_return = self._test_return_tracker.get_mean_return()
            test_ep_len = self._test_return_tracker.get_mean_ep_len()
            test_info = {
                "mean_return": test_return.item(),
                "mean_ep_len": test_ep_len.item(),
                "num_eps": self._test_return_tracker.get_episodes()
            }
        return test_info

    def _step_env(self, action):
        obs, r, done, info = self._env.step(action)
        return obs, r, done, info

    def _record_data_pre_step(self, obs, info, action, action_info):
        self._exp_buffer.record("obs", obs)
        self._exp_buffer.record("action", action)
        
        if (self._need_normalizer_update()):
            self._obs_norm.record(obs)
        return

    def _record_data_post_step(self, next_obs, r, done, next_info):
        self._exp_buffer.record("next_obs", next_obs)
        self._exp_buffer.record("reward", r)
        self._exp_buffer.record("done", done)
        return

    def _reset_done_envs(self, done):
        done_indices = (done != base_env.DoneFlags.NULL.value).nonzero(as_tuple=False)
        env_ids = torch.flatten(done_indices)
        obs, info = self._reset_envs(env_ids)
        return obs, info

    def _reset_envs(self, env_ids=None):
        obs, info = self._env.reset(env_ids)
        return obs, info

    def _need_normalizer_update(self):
        return self._sample_count < self._normalizer_samples

    def _update_normalizers(self):
        self._obs_norm.update()
        return

    def _build_train_data(self):
        return dict()

    def _compute_succ_val(self):
        r_succ = self._env.get_reward_succ()
        val_succ = r_succ / (1.0 - self._discount)
        return val_succ
    
    def _compute_fail_val(self):
        r_fail = self._env.get_reward_fail()
        val_fail = r_fail / (1.0 - self._discount)
        return val_fail

    def _log_train_info(self, train_info, test_info, env_diag_info, start_time):
        wall_time = (time.time() - start_time) / (60 * 60) # store time in hours
        self._logger.log("Iteration", self._iter, collection="1_Info")
        self._logger.log("Wall_Time", wall_time, collection="1_Info")
        self._logger.log("Samples", self._sample_count, collection="1_Info")
        
        test_return = test_info["mean_return"]
        test_ep_len = test_info["mean_ep_len"]
        test_eps = test_info["num_eps"]
        test_eps = mp_util.reduce_sum(test_eps)

        self._logger.log("Test_Return", test_return, collection="0_Main")
        self._logger.log("Test_Episode_Length", test_ep_len, collection="0_Main", quiet=True)
        self._logger.log("Test_Episodes", test_eps, collection="1_Info", quiet=True)

        train_return = train_info.pop("mean_return")
        train_ep_len = train_info.pop("mean_ep_len")
        train_eps = train_info.pop("num_eps")
        train_eps = mp_util.reduce_sum(train_eps)

        self._logger.log("Train_Return", train_return, collection="0_Main")
        self._logger.log("Train_Episode_Length", train_ep_len, collection="0_Main", quiet=True)
        self._logger.log("Train_Episodes", train_eps, collection="1_Info", quiet=True)

        for k, v in train_info.items():
            val_name = k.title()
            if torch.is_tensor(v):
                v = v.item()
            self._logger.log(val_name, v)

        for k, v in env_diag_info.items():
            val_name = k.title()
            if torch.is_tensor(v):
                v = v.item()
            self._logger.log(val_name, v, collection="2_Env", quiet=True)
        
        obs_norm_mean = self._obs_norm.get_mean()
        obs_norm_std = self._obs_norm.get_std()

        if (obs_norm_mean.dtype == torch.float32):
            obs_norm_mean = torch.mean(torch.abs(obs_norm_mean)).item()
            obs_norm_std = torch.mean(obs_norm_std).item()
            self._logger.log("Obs_Norm_Mean", obs_norm_mean, quiet=True)
            self._logger.log("Obs_Norm_Std", obs_norm_std, quiet=True)
        
        return
    
    def _compute_action_bound_loss(self, norm_a_dist):
        loss = None
        action_space = self._env.get_action_space()

        if (isinstance(action_space, spaces.Box)):
            a_low = action_space.low
            a_high = action_space.high
            valid_bounds = np.all(np.isfinite(a_low)) and np.all(np.isfinite(a_high))

            if (valid_bounds):
                assert(isinstance(norm_a_dist, distribution_gaussian_diag.DistributionGaussianDiag))
                # assume that actions have been normalized between [-1, 1]
                bound_min = -1
                bound_max = 1
                violation_min = torch.clamp_max(norm_a_dist.mode - bound_min, 0.0)
                violation_max = torch.clamp_min(norm_a_dist.mode - bound_max, 0)
                violation = torch.sum(torch.square(violation_min), dim=-1) \
                            + torch.sum(torch.square(violation_max), dim=-1)
                loss = violation

        return loss

    def _output_train_model(self, iter, out_model_file, int_out_dir):
        self.save(out_model_file)

        if (int_out_dir != ""):
            int_model_file = os.path.join(int_out_dir, "model_{:010d}.pt".format(iter))
            self.save(int_model_file)
        return
    
    @abc.abstractmethod
    def _build_model(self, config):
        return
    
    @abc.abstractmethod
    def _build_optimizer(self, config):
        return

    @abc.abstractmethod
    def _get_exp_buffer_length(self):
        return 0
    
    @abc.abstractmethod
    def _sync_optimizer(self):
        return

    @abc.abstractmethod
    def _decide_action(self, obs, info):
        a = None
        a_info = dict()
        return a, a_info
    
    @abc.abstractmethod
    def _update_model(self):
        return