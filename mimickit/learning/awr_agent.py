import numpy as np
import torch

import envs.base_env as base_env
import learning.base_agent as base_agent
import learning.awr_model as awr_model
import learning.mp_optimizer as mp_optimizer
import learning.rl_util as rl_util
import util.mp_util as mp_util
import util.torch_util as torch_util

class AWRAgent(base_agent.BaseAgent):
    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        return

    def _load_params(self, config):
        super()._load_params(config)
        
        self._actor_epochs = config["actor_epochs"]
        self._actor_batch_size = config["actor_batch_size"]
        self._critic_epochs = config["critic_epochs"]
        self._critic_batch_size = config["critic_batch_size"]
        
        self._td_lambda = config["td_lambda"]
        self._awr_temp = config["awr_temp"]
        self._a_weight_clip = config["a_weight_clip"]

        self._action_bound_weight = config["action_bound_weight"]
        self._action_entropy_weight = config["action_entropy_weight"]
        self._action_reg_weight = config["action_reg_weight"]

        self._critic_eval_batch_size = int(config.get("critic_eval_batch_size", 0))
        
        self._exp_anneal_samples = config.get("exp_anneal_samples", np.inf)
        self._exp_prob_beg = config.get("exp_prob_beg", 1.0)
        self._exp_prob_end = config.get("exp_prob_end", 1.0)
        return

    def _build_model(self, config):
        model_config = config["model"]
        self._model = awr_model.AWRModel(model_config, self._env)
        return
    
    def _build_optimizer(self, config):
        actor_config = config["actor_optimizer"]
        actor_params = list(self._model.get_actor_params())
        actor_params = [p for p in actor_params if p.requires_grad]
        self._actor_optimizer = mp_optimizer.MPOptimizer(actor_config, actor_params)
        
        critic_config = config["critic_optimizer"]
        critic_params = list(self._model.get_critic_params())
        critic_params = [p for p in critic_params if p.requires_grad]
        self._critic_optimizer = mp_optimizer.MPOptimizer(critic_config, critic_params)
        return
    
    def _sync_optimizer(self):
        self._actor_optimizer.sync()
        self._critic_optimizer.sync()
        return
    
    def _get_exp_buffer_length(self):
        return self._steps_per_iter
    
    def _init_iter(self):
        super()._init_iter()
        self._exp_buffer.reset()
        return

    def _decide_action(self, obs, info):
        norm_obs = self._obs_norm.normalize(obs)
        norm_action_dist = self._model.eval_actor(norm_obs)

        if (self._mode == base_agent.AgentMode.TRAIN):
            norm_a_rand = norm_action_dist.sample()
            norm_a_mode = norm_action_dist.mode

            exp_prob = self._get_exp_prob()
            exp_prob = torch.full([norm_a_rand.shape[0], 1], exp_prob, device=self._device, dtype=torch.float)
            rand_action_mask = torch.bernoulli(exp_prob)
            norm_a = torch.where(rand_action_mask == 1.0, norm_a_rand, norm_a_mode)
            rand_action_mask = rand_action_mask.squeeze(-1)

        elif (self._mode == base_agent.AgentMode.TEST):
            norm_a = norm_action_dist.mode
            rand_action_mask = torch.zeros_like(norm_a[..., 0])
        else:
            assert(False), "Unsupported agent mode: {}".format(self._mode)
            
        norm_a = norm_a.detach()
        a = self._a_norm.unnormalize(norm_a)

        a_info = {
            "rand_action_mask": rand_action_mask
        }
        return a, a_info

    def _record_data_pre_step(self, obs, info, action, action_info):
        super()._record_data_pre_step(obs, info, action, action_info)
        self._exp_buffer.record("rand_action_mask", action_info["rand_action_mask"])
        return
    
    def _build_train_data(self):
        self.eval()
        
        obs = self._exp_buffer.get_data("obs")
        next_obs = self._exp_buffer.get_data("next_obs")
        r = self._exp_buffer.get_data("reward")
        done = self._exp_buffer.get_data("done")
        rand_action_mask = self._exp_buffer.get_data("rand_action_mask")
        
        norm_next_obs = self._obs_norm.normalize(next_obs)
        next_critic_inputs = {"obs": norm_next_obs}
        next_vals = torch_util.eval_minibatch(self._model.eval_critic, next_critic_inputs, self._critic_eval_batch_size)
        next_vals = next_vals.squeeze(-1).detach()

        succ_val = self._compute_succ_val()
        succ_mask = (done == base_env.DoneFlags.SUCC.value)
        next_vals[succ_mask] = succ_val

        fail_val = self._compute_fail_val()
        fail_mask = (done == base_env.DoneFlags.FAIL.value)
        next_vals[fail_mask] = fail_val

        new_vals = rl_util.compute_td_lambda_return(r, next_vals, done, self._discount, self._td_lambda)

        norm_obs = self._obs_norm.normalize(obs)
        critic_inputs = {"obs": norm_obs}
        vals = torch_util.eval_minibatch(self._model.eval_critic, critic_inputs, self._critic_eval_batch_size)
        vals = vals.squeeze(-1).detach()
        adv = new_vals - vals
        
        rand_action_mask = (rand_action_mask == 1.0).flatten()
        adv_flat = adv.flatten()
        rand_action_adv = adv_flat[rand_action_mask]
        adv_mean, adv_std = mp_util.calc_mean_std(rand_action_adv)
        norm_adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-5)
        
        a_weight = torch.exp(norm_adv / self._awr_temp)
        a_weight = torch.clamp_max(a_weight, self._a_weight_clip)

        self._exp_buffer.set_data("tar_val", new_vals)
        self._exp_buffer.set_data("a_weight", a_weight)
        
        a_weight_mean = torch.mean(a_weight)
        a_weight_min = torch.min(a_weight)
        a_weight_max = torch.max(a_weight)

        info = {
            "adv_mean": adv_mean,
            "adv_std": adv_std,
            "a_weight_mean": a_weight_mean,
            "a_weight_min": a_weight_min,
            "a_weight_max": a_weight_max
        }
        return info
    
    def _get_exp_prob(self):
        if (np.isfinite(self._exp_anneal_samples)):
            samples = self._sample_count
            l = float(samples) / self._exp_anneal_samples
            l = np.clip(l, 0.0, 1.0)
            prob = (1.0 - l) * self._exp_prob_beg + l * self._exp_prob_end
        else:
            prob = self._exp_prob_beg
        return prob
    
    def _update_model(self):
        self.train()
        
        num_envs = self.get_num_envs()
        num_samples = self._exp_buffer.get_sample_count()

        critic_batch_size = int(np.ceil(self._critic_batch_size * num_envs))
        num_critic_batches = int(np.ceil(float(num_samples) / critic_batch_size))
        num_critic_steps = num_critic_batches * self._critic_epochs
        critic_info = self._update_critic(critic_batch_size, num_critic_steps)
        
        actor_batch_size = int(np.ceil(self._actor_batch_size * num_envs))
        num_actor_batches = int(np.ceil(float(num_samples) / actor_batch_size))
        num_actor_steps = num_actor_batches * self._actor_epochs
        actor_info = self._update_actor(actor_batch_size, num_actor_steps)
        
        train_info = {**critic_info, **actor_info}
        return train_info
    
    def _update_critic(self, batch_size, steps):
        info = dict()

        for i in range(steps):
            batch = self._exp_buffer.sample(batch_size)
            loss_info = None
            def loss_factory():
                nonlocal loss_info
                loss_info = self._compute_critic_loss(batch)
                return loss_info["critic_loss"]
            self._critic_optimizer.step(loss_factory)

            torch_util.add_torch_dict(loss_info, info)
        
        torch_util.scale_torch_dict(1.0 / steps, info)
        return info

    def _update_actor(self, batch_size, num_steps):
        info = dict()

        for i in range(num_steps):
            batch = self._exp_buffer.sample(batch_size)
            loss_info = None
            def loss_factory():
                nonlocal loss_info
                loss_info = self._compute_actor_loss(batch)
                return loss_info["actor_loss"]
            self._actor_optimizer.step(loss_factory)

            torch_util.add_torch_dict(loss_info, info)
        
        torch_util.scale_torch_dict(1.0 / num_steps, info)
        return info
    
    def _compute_critic_loss(self, batch):
        norm_obs = self._obs_norm.normalize(batch["obs"])
        tar_val = batch["tar_val"]
        pred = self._model.eval_critic(norm_obs)
        pred = pred.squeeze(-1)

        diff = tar_val - pred
        loss = torch.mean(torch.square(diff))

        info = {
            "critic_loss": loss
        }
        return info

    def _compute_actor_loss(self, batch):
        norm_obs = self._obs_norm.normalize(batch["obs"])
        norm_a = self._a_norm.normalize(batch["action"])
        a_weight = batch["a_weight"]
        rand_action_mask = batch["rand_action_mask"]

        # loss should only be computed using samples with random actions
        rand_action_mask = (rand_action_mask == 1.0)
        norm_obs = norm_obs[rand_action_mask]
        norm_a = norm_a[rand_action_mask]
        a_weight = a_weight[rand_action_mask]

        a_dist = self._model.eval_actor(norm_obs)
        a_logp = a_dist.log_prob(norm_a)
        
        actor_loss = a_weight * a_logp
        actor_loss = -torch.mean(actor_loss)
        
        info = {
            "actor_loss": actor_loss
        }

        if (self._action_bound_weight != 0):
            action_bound_loss = self._compute_action_bound_loss(a_dist)
            if (action_bound_loss is not None):
                action_bound_loss = torch.mean(action_bound_loss)
                actor_loss += self._action_bound_weight * action_bound_loss
                info["action_bound_loss"] = action_bound_loss.detach()

        if (self._action_entropy_weight != 0):
            action_entropy = a_dist.entropy()
            action_entropy = torch.mean(action_entropy)
            actor_loss += -self._action_entropy_weight * action_entropy
            info["action_entropy"] = action_entropy.detach()
        
        if (self._action_reg_weight != 0):
            action_reg_loss = a_dist.param_reg()
            action_reg_loss = torch.mean(action_reg_loss)
            actor_loss += self._action_reg_weight * action_reg_loss
            info["action_reg_loss"] = action_reg_loss.detach()
        
        return info

    def _log_train_info(self, train_info, test_info, env_diag_info, start_time):
        super()._log_train_info(train_info, test_info, env_diag_info, start_time)
        self._logger.log("Exp_Prob", self._get_exp_prob())
        return