import torch

import envs.amp_env as amp_env
import envs.char_env as char_env

class ASEEnv(amp_env.AMPEnv):
    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        self._default_reset_prob = env_config["default_reset_prob"]

        super().__init__(env_config=env_config, engine_config=engine_config, 
                         num_envs=num_envs, device=device, visualize=visualize,
                         record_video=record_video)
        return
    
    def _reset_char(self, env_ids):
        super()._reset_char(env_ids)
        
        n = env_ids.shape[0]
        if (n > 0):
            rand_val = torch.rand(n, device=self._device)
            mask = rand_val < self._default_reset_prob
            default_reset_ids = env_ids[mask]
            char_env.CharEnv._reset_char(self, default_reset_ids)

        return
