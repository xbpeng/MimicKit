import torch
import numpy as np

import envs.amp_env as amp_env
import engines.engine as engine
import util.torch_util as torch_util


class TaskSteeringEnv(amp_env.AMPEnv):
    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        self._rand_tar_dir = env_config.get("rand_tar_dir", True)
        self._rand_face_dir = env_config.get("rand_face_dir", True)
        self._tar_speed_min = env_config["tar_speed_min"]
        self._tar_speed_max = env_config["tar_speed_max"]
        self._tar_change_time_min = env_config["tar_change_time_min"]
        self._tar_change_time_max = env_config["tar_change_time_max"]

        self._reward_steering_tar_w = float(env_config["reward_steering_tar_w"])
        self._reward_steering_face_w = float(env_config["reward_steering_face_w"])

        self._reward_steering_vel_scale = float(env_config["reward_steering_vel_scale"])

        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize,
                         record_video=record_video)
        return
    
    def _build_envs(self, config, num_envs):
        if (self._visualize):
            self._tar_marker_ids = []
            self._face_marker_ids = []

        super()._build_envs(config, num_envs)
        return
    
    def _build_env(self, env_id, config):
        super()._build_env(env_id, config)
        
        if (self._visualize):
            tar_marker_id, face_marker_id = self._build_markers(env_id)

            if (env_id == 0):
                self._tar_marker_ids.append(tar_marker_id)
                self._face_marker_ids.append(face_marker_id)
            else:
                tar_marker_id0 = self._tar_marker_ids[0]
                face_marker_id0 = self._face_marker_ids[0]

                assert(tar_marker_id0 == tar_marker_id)
                assert(face_marker_id0 == face_marker_id)
        return
    
    def _build_markers(self, env_id):
        asset_file = "data/assets/objects/steering_marker.xml"

        tar_marker_id = self._engine.create_obj(env_id=env_id, 
                                             obj_type=engine.ObjType.rigid,
                                             asset_file=asset_file, 
                                             name="tar_marker",
                                             is_visual=True,
                                             enable_self_collisions=False,
                                             fix_root=True,
                                             color=[0.8, 0.0, 0.0])

        face_marker_id = self._engine.create_obj(env_id=env_id, 
                                             obj_type=engine.ObjType.rigid,
                                             asset_file=asset_file, 
                                             name="face_marker",
                                             is_visual=True,
                                             enable_self_collisions=False,
                                             fix_root=True,
                                             color=[0.0, 0.8, 0.0])

        return tar_marker_id, face_marker_id

    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)

        num_envs = self.get_num_envs()
        self._tar_speed = torch.ones([num_envs], device=self._device, dtype=torch.float)
        self._tar_dir = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)
        self._face_dir = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)
        self._tar_change_times = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._prev_root_pos = torch.zeros([num_envs, 3], device=self._device, dtype=torch.float)
        
        self._tar_dir[..., 0] = 1.0
        self._face_dir[..., 0] = 1.0
        return
    
    def _get_tar_marker_id(self):
        return self._tar_marker_ids[0]
    
    def _get_face_marker_id(self):
        return self._face_marker_ids[0]

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        self._record_char_root()
        return
    
    def _record_char_root(self):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        self._prev_root_pos[:] = root_pos
        return

    def _update_marker(self, env_ids):
        tar_dist_min = 1.0
        tar_dist_max = 1.5

        char_id = self._get_char_id()
        tar_marker_id = self._get_tar_marker_id()
        face_marker_id = self._get_face_marker_id()
        
        root_pos = self._engine.get_root_pos(char_id)
        root_pos = root_pos[env_ids]
        tar_speed = self._tar_speed[env_ids]
        tar_dir = self._tar_dir[env_ids]
        face_dir = self._face_dir[env_ids]

        tar_dist = (tar_speed - self._tar_speed_min) / (self._tar_speed_max - self._tar_speed_min)
        tar_dist = (tar_dist_max - tar_dist_min) * tar_dist + tar_dist_min
        tar_dist = tar_dist.unsqueeze(-1)

        marker_pos = root_pos.clone()
        marker_pos[..., 0:2] += tar_dist * tar_dir
        marker_pos[..., 2] = 0.0

        tar_theta = torch.atan2(tar_dir[..., 1], tar_dir[..., 0])
        tar_axis = torch.zeros_like(root_pos)
        tar_axis[..., -1] = 1.0
        marker_rot = torch_util.axis_angle_to_quat(tar_axis, tar_theta)
        
        face_marker_pos = root_pos.clone()
        face_marker_pos[..., 0:2] += tar_dist_min * face_dir
        face_marker_pos[..., 2] = 0.01

        face_theta = torch.atan2(face_dir[..., 1], face_dir[..., 0])
        face_axis = torch.zeros_like(root_pos)
        face_axis[..., -1] = 1.0
        face_marker_rot = torch_util.axis_angle_to_quat(tar_axis, face_theta)

        self._engine.set_root_pos(env_ids, tar_marker_id, marker_pos)
        self._engine.set_root_rot(env_ids, tar_marker_id, marker_rot)
        self._engine.set_root_vel(env_ids, tar_marker_id, 0.0)
        self._engine.set_root_ang_vel(env_ids, tar_marker_id, 0.0)

        self._engine.set_root_pos(env_ids, face_marker_id, face_marker_pos)
        self._engine.set_root_rot(env_ids, face_marker_id, face_marker_rot)
        self._engine.set_root_vel(env_ids, face_marker_id, 0.0)
        self._engine.set_root_ang_vel(env_ids, face_marker_id, 0.0)
        return
    
    def _update_task(self):
        reset_task_mask = self._time_buf >= self._tar_change_times
        rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()

        if len(rest_env_ids) > 0:
            self._reset_task(rest_env_ids)
        return
    
    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        
        if (len(env_ids) > 0):
            self._reset_task(env_ids)
        return

    def _reset_task(self, env_ids):
        self._reset_tar(env_ids)

        n = env_ids.shape[0]
        rand_dt = torch.rand(n, device=self._device)
        rand_dt = (self._tar_change_time_max - self._tar_change_time_min) * rand_dt + self._tar_change_time_min
        self._tar_change_times[env_ids] = self._time_buf[env_ids] + rand_dt
        
        if (self._visualize):
            self._update_marker(env_ids)
        return

    def _reset_tar(self, env_ids):
        n = env_ids.shape[0]

        if (self._rand_tar_dir):
            rand_theta = 2 * np.pi * torch.rand(n, device=self._device) - np.pi
        else:
            rand_theta = torch.zeros(n, device=self._device)

        if (self._rand_face_dir):
            rand_face_theta = 2 * np.pi * torch.rand(n, device=self._device) - np.pi
        else:
            rand_face_theta = rand_theta.clone()

        tar_dir = torch.stack([torch.cos(rand_theta), torch.sin(rand_theta)], dim=-1)
        tar_speed = (self._tar_speed_max - self._tar_speed_min) * torch.rand(n, device=self._device) + self._tar_speed_min
        face_tar_dir = torch.stack([torch.cos(rand_face_theta), torch.sin(rand_face_theta)], dim=-1)

        self._tar_speed[env_ids] = tar_speed
        self._tar_dir[env_ids] = tar_dir
        self._face_dir[env_ids] = face_tar_dir
        return
    
    def _compute_obs(self, env_ids=None):
        obs = super()._compute_obs(env_ids)
        
        char_id = self._get_char_id()
        root_rot = self._engine.get_root_rot(char_id)
        tar_dir = self._tar_dir
        tar_speed = self._tar_speed
        face_dir = self._face_dir

        if (env_ids is not None):
            root_rot = root_rot[env_ids]
            tar_dir = tar_dir[env_ids]
            tar_speed = tar_speed[env_ids]
            face_dir = face_dir[env_ids]
        
        task_obs = compute_steering_observations(root_rot, tar_dir, tar_speed, face_dir)
        obs = torch.cat([obs, task_obs], dim=-1)

        return obs
    
    def _update_reward(self):
        char_id = self._get_char_id()
        char_root_pos = self._engine.get_root_pos(char_id)
        char_root_rot = self._engine.get_root_rot(char_id)

        self._reward_buf[:] = compute_steering_reward(root_pos=char_root_pos,
                                                      prev_root_pos=self._prev_root_pos,
                                                      root_rot=char_root_rot,
                                                      tar_dir=self._tar_dir, 
                                                      tar_speed=self._tar_speed,
                                                      face_dir=self._face_dir,
                                                      dt=self._engine.get_timestep(),
                                                      vel_err_scale=self._reward_steering_vel_scale,
                                                      tar_reward_w=self._reward_steering_tar_w,
                                                      face_reward_w=self._reward_steering_face_w)
        return
    
    def _update_misc(self):
        super()._update_misc()
        self._update_task()
        return

    

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_steering_observations(root_rot, tar_dir, tar_speed, face_dir):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
    
    tar_dir3d = torch.cat([tar_dir, torch.zeros_like(tar_dir[..., 0:1])], dim=-1)
    heading_rot = torch_util.calc_heading_quat_inv(root_rot)
    
    local_tar_dir = torch_util.quat_rotate(heading_rot, tar_dir3d)
    local_tar_dir = local_tar_dir[..., 0:2]
    tar_speed = tar_speed.unsqueeze(-1)
    
    face_dir3d = torch.cat([face_dir, torch.zeros_like(face_dir[..., 0:1])], dim=-1)
    local_face_dir = torch_util.quat_rotate(heading_rot, face_dir3d)
    local_face_dir = local_face_dir[..., 0:2]

    obs = torch.cat([local_tar_dir, tar_speed, local_face_dir], dim=-1)

    return obs


@torch.jit.script
def compute_steering_reward(root_pos, prev_root_pos, root_rot, tar_dir, tar_speed, face_dir, dt,
                           vel_err_scale, tar_reward_w, face_reward_w):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float) -> Tensor

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt

    tar_vel = tar_speed.unsqueeze(-1) * tar_dir
    tar_vel_err = tar_vel - root_vel[..., :2]
    tar_vel_err = torch.sum(torch.square(tar_vel_err), dim=-1)

    tar_reward = torch.exp(-vel_err_scale * tar_vel_err)
    
    proj_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    wrong_dir = proj_speed < 0
    tar_reward[wrong_dir] = 0

    heading_rot = torch_util.calc_heading_quat(root_rot)
    char_face_dir = torch.zeros_like(root_pos)
    char_face_dir[..., 0] = 1.0
    char_face_dir = torch_util.quat_rotate(heading_rot, char_face_dir)
    face_err = torch.sum(face_dir * char_face_dir[..., 0:2], dim=-1)
    face_reward = torch.clamp_min(face_err, 0.0)

    reward = tar_reward_w * tar_reward + face_reward_w * face_reward

    return reward