import gymnasium.spaces as spaces
import numpy as np
import torch

import envs.deepmimic_env as deepmimic_env
import util.circular_buffer as circular_buffer
import util.torch_util as torch_util

class AMPEnv(deepmimic_env.DeepMimicEnv):
    def __init__(self, env_config, engine_config, num_envs, device, visualize):
        self._num_disc_obs_steps = env_config["num_disc_obs_steps"]

        super().__init__(env_config=env_config, engine_config=engine_config, num_envs=num_envs, device=device,
                         visualize=visualize)
        return

    def get_disc_obs_space(self):
        disc_obs = self.fetch_disc_obs_demo(1)
        disc_obs_shape = list(disc_obs.shape[1:])
        disc_obs_dtype = torch_util.torch_dtype_to_numpy(disc_obs.dtype)
        disc_obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=disc_obs_shape,
            dtype=disc_obs_dtype,
        )
        return disc_obs_space

    def fetch_disc_obs_demo(self, num_samples):
        motion_ids, motion_times0 = self._sample_motion_times(num_samples)
        disc_obs = self._compute_disc_obs_demo(motion_ids, motion_times0)
        return disc_obs

    def _compute_disc_obs_demo(self, motion_ids, motion_times0):
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, body_pos = self._fetch_disc_demo_data(motion_ids, motion_times0)
        
        if (self._track_global_root()):
            ref_root_pos = torch.zeros_like(root_pos[..., -1, :])
            ref_root_rot = torch.zeros_like(root_rot[..., -1, :])
            ref_root_rot[..., -1] = 1
        else:
            ref_root_pos = root_pos[..., -1, :]
            ref_root_rot = root_rot[..., -1, :]
            
        if (self._has_key_bodies()):
            key_pos = body_pos[..., self._key_body_ids, :]
        else:
            key_pos = torch.zeros([0], device=self._device)
        
        disc_obs = compute_disc_obs(ref_root_pos=ref_root_pos,
                                  ref_root_rot=ref_root_rot,
                                  root_pos=root_pos,
                                  root_rot=root_rot, 
                                  root_vel=root_vel,
                                  root_ang_vel=root_ang_vel,
                                  joint_rot=joint_rot,
                                  dof_vel=dof_vel,
                                  key_pos=key_pos,
                                  global_obs=self._global_obs,
                                  root_height_obs=self._root_height_obs)
        return disc_obs

    def _fetch_disc_demo_data(self, motion_ids, motion_times0):
        num_samples = motion_ids.shape[0]

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_disc_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -self._engine.get_timestep() * torch.arange(0, self._num_disc_obs_steps, device=self._device)
        time_steps = torch.flip(time_steps, dims=[0])
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        
        body_pos, _ = self._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)

        root_pos = torch.reshape(root_pos, shape=[num_samples, self._num_disc_obs_steps, root_pos.shape[-1]])
        root_rot = torch.reshape(root_rot, shape=[num_samples, self._num_disc_obs_steps, root_rot.shape[-1]])
        root_vel = torch.reshape(root_vel, shape=[num_samples, self._num_disc_obs_steps, root_vel.shape[-1]])
        root_ang_vel = torch.reshape(root_ang_vel, shape=[num_samples, self._num_disc_obs_steps, root_ang_vel.shape[-1]])
        joint_rot = torch.reshape(joint_rot, shape=[num_samples, self._num_disc_obs_steps, joint_rot.shape[-2], joint_rot.shape[-1]])
        dof_vel = torch.reshape(dof_vel, shape=[num_samples, self._num_disc_obs_steps, dof_vel.shape[-1]])
        body_pos = torch.reshape(body_pos, shape=[num_samples, self._num_disc_obs_steps, body_pos.shape[-2], body_pos.shape[-1]])
        
        return root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, body_pos

    def _build_data_buffers(self):
        super()._build_data_buffers()
        self._build_disc_obs_buffers()
        self._info["disc_obs"] = self._disc_obs_buf
        return

    def _build_disc_obs_buffers(self):
        num_envs = self.get_num_envs()
        n = self._num_disc_obs_steps
        
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        root_vel = self._engine.get_root_vel(char_id)
        root_ang_vel = self._engine.get_root_ang_vel(char_id)
        dof_pos = self._engine.get_dof_pos(char_id)
        dof_vel = self._engine.get_dof_vel(char_id)
        body_pos = self._engine.get_body_pos(char_id)
        
        joint_rot = self._kin_char_model.dof_to_rot(dof_pos)

        self._disc_hist_root_pos = circular_buffer.CircularBuffer(batch_size=num_envs,
                                                                 buffer_len=n, 
                                                                 shape=root_pos.shape[1:],
                                                                 dtype=root_pos.dtype,
                                                                 device=self._device)

        self._disc_hist_root_rot = circular_buffer.CircularBuffer(batch_size=num_envs,
                                                                 buffer_len=n,
                                                                 shape=root_rot.shape[1:],
                                                                 dtype=root_rot.dtype,
                                                                 device=self._device)

        self._disc_hist_root_vel = circular_buffer.CircularBuffer(batch_size=num_envs,
                                                                 buffer_len=n,
                                                                 shape=root_vel.shape[1:],
                                                                 dtype=root_vel.dtype,
                                                                 device=self._device)
        
        self._disc_hist_root_ang_vel = circular_buffer.CircularBuffer(batch_size=num_envs,
                                                                     buffer_len=n,
                                                                     shape=root_ang_vel.shape[1:],
                                                                     dtype=root_ang_vel.dtype,
                                                                     device=self._device)
        
        self._disc_hist_joint_rot = circular_buffer.CircularBuffer(batch_size=num_envs,
                                                                  buffer_len=n,
                                                                  shape=joint_rot.shape[1:],
                                                                  dtype=joint_rot.dtype,
                                                                  device=self._device)

        self._disc_hist_dof_vel = circular_buffer.CircularBuffer(batch_size=num_envs,
                                                                buffer_len=n,
                                                                shape=dof_vel.shape[1:],
                                                                dtype=dof_vel.dtype,
                                                                device=self._device)
        
        self._disc_hist_body_pos = circular_buffer.CircularBuffer(batch_size=num_envs,
                                                                 buffer_len=n,
                                                                 shape=body_pos.shape[1:],
                                                                 dtype=body_pos.dtype,
                                                                 device=self._device)

        disc_obs_space = self.get_disc_obs_space()
        disc_obs_dtype = torch_util.numpy_dtype_to_torch(disc_obs_space.dtype)
        self._disc_obs_buf = torch.zeros([num_envs] + list(disc_obs_space.shape), device=self._device, dtype=disc_obs_dtype)
        return

    def _update_misc(self):
        super()._update_misc()
        self._update_disc_hist()
        return
    
    def _update_disc_hist(self):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        root_vel = self._engine.get_root_vel(char_id)
        root_ang_vel = self._engine.get_root_ang_vel(char_id)
        dof_pos = self._engine.get_dof_pos(char_id)
        dof_vel = self._engine.get_dof_vel(char_id)
        body_pos = self._engine.get_body_pos(char_id)

        self._disc_hist_root_pos.push(root_pos)
        self._disc_hist_root_rot.push(root_rot)
        self._disc_hist_root_vel.push(root_vel)
        self._disc_hist_root_ang_vel.push(root_ang_vel)
        self._disc_hist_dof_vel.push(dof_vel)
        self._disc_hist_body_pos.push(body_pos)

        # Validate dof_pos before passing to dof_to_rot
        if not isinstance(dof_pos, torch.Tensor):
            raise TypeError(
                f"_update_disc_hist: dof_pos expected torch.Tensor, got {type(dof_pos).__name__}"
            )

        joint_rot = self._kin_char_model.dof_to_rot(dof_pos)
        self._disc_hist_joint_rot.push(joint_rot)
        return
    
    def _update_ref_motion(self):
        if (self._enable_ref_char()):
            super()._update_ref_motion()
        return
    
    def _update_observations(self, env_ids=None):
        super()._update_observations(env_ids)

        if (env_ids is None or len(env_ids) > 0):
            self._update_disc_obs(env_ids)
        return

    def _update_disc_obs(self, env_ids=None):
        root_pos = self._disc_hist_root_pos.get_all()
        root_rot = self._disc_hist_root_rot.get_all()
        root_vel = self._disc_hist_root_vel.get_all()
        root_ang_vel = self._disc_hist_root_ang_vel.get_all()
        joint_rot = self._disc_hist_joint_rot.get_all()
        dof_vel = self._disc_hist_dof_vel.get_all()
        body_pos = self._disc_hist_body_pos.get_all()
        
        if (env_ids is not None):
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]
            root_vel = root_vel[env_ids]
            root_ang_vel = root_ang_vel[env_ids]
            joint_rot = joint_rot[env_ids]
            dof_vel = dof_vel[env_ids]
            body_pos = body_pos[env_ids]
        
        if (self._track_global_root()):
            ref_root_pos = torch.zeros_like(root_pos[..., -1, :])
            ref_root_rot = torch.zeros_like(root_rot[..., -1, :])
            ref_root_rot[..., -1] = 1
        else:
            ref_root_pos = root_pos[..., -1, :]
            ref_root_rot = root_rot[..., -1, :]
            
        if (self._has_key_bodies()):
            key_pos = body_pos[..., self._key_body_ids, :]
        else:
            key_pos = torch.zeros([0], device=self._device)

        disc_obs = compute_disc_obs(ref_root_pos=ref_root_pos,
                                  ref_root_rot=ref_root_rot,
                                  root_pos=root_pos,
                                  root_rot=root_rot, 
                                  root_vel=root_vel,
                                  root_ang_vel=root_ang_vel,
                                  joint_rot=joint_rot,
                                  dof_vel=dof_vel,
                                  key_pos=key_pos,
                                  global_obs=self._global_obs,
                                  root_height_obs=self._root_height_obs)

        if (env_ids is None):
            self._disc_obs_buf[:] = disc_obs
        else:
            self._disc_obs_buf[env_ids] = disc_obs

        return

    def _update_done(self):
        motion_times = self._get_motion_times()
        motion_len = self._motion_lib.get_motion_length(self._motion_ids)
        motion_len_term = torch.full(motion_len.shape, False, device=motion_len.device, dtype=torch.bool)

        track_root = self._track_global_root()
        
        char_id = self._get_char_id()
        root_rot = self._engine.get_root_rot(char_id)
        body_pos = self._engine.get_body_pos(char_id)
        ground_contact_forces = self._engine.get_ground_contact_forces(char_id)

        self._done_buf[:] = deepmimic_env.compute_done(done_buf=self._done_buf,
                                         time=self._time_buf, 
                                         ep_len=self._episode_length, 
                                         root_rot=root_rot,
                                         body_pos=body_pos,
                                         tar_root_rot=self._ref_root_rot,
                                         tar_body_pos=self._ref_body_pos,
                                         ground_contact_force=ground_contact_forces,
                                         contact_body_ids=self._contact_body_ids,
                                         pose_termination=self._pose_termination,
                                         pose_termination_dist=self._pose_termination_dist,
                                         global_obs=self._global_obs,
                                         enable_early_termination=self._enable_early_termination,
                                         motion_times=motion_times,
                                         motion_len=motion_len,
                                         motion_len_term=motion_len_term,
                                         track_root=track_root)
        return

    def _update_reward(self):
        return
    
    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)

        if (len(env_ids) > 0):
            self._reset_disc_hist(env_ids)
        return

    def _reset_disc_hist(self, env_ids):
        motion_ids = self._motion_ids[env_ids]
        motion_times0 = self._get_motion_times(env_ids)
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, body_pos = self._fetch_disc_demo_data(motion_ids, motion_times0)
        
        self._disc_hist_root_pos.fill(env_ids, root_pos)
        self._disc_hist_root_rot.fill(env_ids, root_rot)
        self._disc_hist_root_vel.fill(env_ids, root_vel)
        self._disc_hist_root_ang_vel.fill(env_ids, root_ang_vel)
        self._disc_hist_joint_rot.fill(env_ids, joint_rot)
        self._disc_hist_dof_vel.fill(env_ids, dof_vel)
        self._disc_hist_body_pos.fill(env_ids, body_pos)
        return

    
@torch.jit.script
def compute_disc_vel_obs(ref_root_rot, root_vel, root_ang_vel, dof_vel, global_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor

    if (not global_obs):
        heading_inv_rot = torch_util.calc_heading_quat_inv(ref_root_rot)
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, root_vel.shape[1], 1))
        heading_inv_rot_flat = heading_inv_rot_expand.reshape((heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1], 
                                                               heading_inv_rot_expand.shape[2]))
        
        root_vel_obs_flat = torch.reshape(root_vel, [root_vel.shape[0] * root_vel.shape[1], root_vel.shape[2]])
        root_vel_obs_flat = torch_util.quat_rotate(heading_inv_rot_flat, root_vel_obs_flat)
        root_vel_obs = torch.reshape(root_vel_obs_flat, root_vel.shape)
        
        root_ang_vel_obs_flat = torch.reshape(root_ang_vel, [root_ang_vel.shape[0] * root_ang_vel.shape[1], root_ang_vel.shape[2]])
        root_ang_vel_obs_flat = torch_util.quat_rotate(heading_inv_rot_flat, root_ang_vel_obs_flat)
        root_ang_vel_obs = torch.reshape(root_ang_vel_obs_flat, root_ang_vel.shape)
    else:
        root_vel_obs = root_vel
        root_ang_vel_obs = root_ang_vel

    obs = [root_vel_obs, root_ang_vel_obs, dof_vel]
    obs = torch.cat(obs, dim=-1)

    return obs

@torch.jit.script
def compute_disc_obs(ref_root_pos, ref_root_rot, root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos, global_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor

    pos_obs = deepmimic_env.compute_tar_obs(ref_root_pos=ref_root_pos, 
                                               ref_root_rot=ref_root_rot, 
                                               root_pos=root_pos, 
                                               root_rot=root_rot, 
                                               joint_rot=joint_rot, 
                                               key_pos=key_pos,
                                               global_obs=global_obs, 
                                               root_height_obs=root_height_obs)

    vel_obs = compute_disc_vel_obs(ref_root_rot=ref_root_rot, 
                                  root_vel=root_vel, 
                                  root_ang_vel=root_ang_vel, 
                                  dof_vel=dof_vel,
                                  global_obs=global_obs)

    disc_obs = torch.cat([pos_obs, vel_obs], dim=-1)
    disc_obs = torch.reshape(disc_obs, [disc_obs.shape[0], -1])

    return disc_obs