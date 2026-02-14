import torch

import envs.amp_env as amp_env
import envs.deepmimic_env as deepmimic_env
import util.torch_util as torch_util

class ADDEnv(amp_env.AMPEnv):
    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        super().__init__(env_config=env_config, engine_config=engine_config, 
                         num_envs=num_envs, device=device, visualize=visualize,
                         record_video=record_video)
        return
    
    def _build_disc_obs_buffers(self):
        super()._build_disc_obs_buffers()
        self._disc_obs_demo_buf = torch.zeros_like(self._disc_obs_buf)
        return

    def _build_data_buffers(self):
        super()._build_data_buffers()
        self._info["disc_obs_demo"] = self._disc_obs_demo_buf
        return

    def _update_observations(self, env_ids=None):
        super()._update_observations(env_ids)
        if (env_ids is None or len(env_ids) > 0):
            self._update_disc_obs_demo(env_ids)
        return
    
    def _update_ref_motion(self):
        deepmimic_env.DeepMimicEnv._update_ref_motion(self)
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

        disc_obs = compute_disc_obs(root_pos=root_pos,
                                  root_rot=root_rot, 
                                  root_vel=root_vel,
                                  root_ang_vel=root_ang_vel,
                                  joint_rot=joint_rot,
                                  dof_vel=dof_vel,
                                  body_pos=body_pos,
                                  global_obs=self._global_obs)

        if (env_ids is None):
            self._disc_obs_buf[:] = disc_obs
        else:
            self._disc_obs_buf[env_ids] = disc_obs

        return
    
    def _update_disc_obs_demo(self, env_ids=None):
        if (env_ids is None):
            motion_ids = self._motion_ids
        else:
            motion_ids = self._motion_ids[env_ids]

        motion_times0 = self._get_motion_times(env_ids)
        disc_obs = self._compute_disc_obs_demo(motion_ids, motion_times0)

        if (env_ids is None):
            self._disc_obs_demo_buf[:] = disc_obs
        else:
            self._disc_obs_demo_buf[env_ids] = disc_obs

        return
    
    def _compute_disc_obs_demo(self, motion_ids, motion_times0):
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, body_pos = self._fetch_disc_demo_data(motion_ids, motion_times0)

        disc_obs = compute_disc_obs(root_pos=root_pos,
                                  root_rot=root_rot, 
                                  root_vel=root_vel,
                                  root_ang_vel=root_ang_vel,
                                  joint_rot=joint_rot,
                                  dof_vel=dof_vel,
                                  body_pos=body_pos,
                                  global_obs=self._global_obs)
        return disc_obs
    
    def _update_done(self):
        deepmimic_env.DeepMimicEnv._update_done(self)
        return

@torch.jit.script
def compute_pos_obs(root_pos, root_rot, joint_rot, body_pos, global_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor

    root_pos_obs = root_pos.detach().clone()
    body_pos = body_pos - root_pos.unsqueeze(-2)

    if (not global_obs):
        root_pos_obs[..., 0:2] = 0.0

        heading_inv_rot = torch_util.calc_heading_quat_inv(root_rot)
        root_rot = torch_util.quat_mul(heading_inv_rot, root_rot)

        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_inv_rot_expand = heading_inv_rot_expand.repeat(1, 1, body_pos.shape[-2], 1)
        heading_inv_rot_expand_flat = heading_inv_rot_expand.reshape((heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1],
                                                                    heading_inv_rot_expand.shape[2], heading_inv_rot_expand.shape[3]))
        body_pos_flat = torch.reshape(body_pos, [body_pos.shape[0] * body_pos.shape[1], body_pos.shape[2], body_pos.shape[3]])
        body_pos_flat = torch_util.quat_rotate(heading_inv_rot_expand_flat, body_pos_flat)
        body_pos = torch.reshape(body_pos_flat, body_pos.shape)

    root_rot_flat = torch.reshape(root_rot, [root_rot.shape[0] * root_rot.shape[1], root_rot.shape[2]])
    root_rot_obs_flat = torch_util.quat_to_tan_norm(root_rot_flat)
    root_rot_obs = torch.reshape(root_rot_obs_flat, [root_rot.shape[0], root_rot.shape[1], root_rot_obs_flat.shape[-1]])

    joint_rot_flat = torch.reshape(joint_rot, [joint_rot.shape[0] * joint_rot.shape[1] * joint_rot.shape[2], joint_rot.shape[3]])
    joint_rot_obs_flat = torch_util.quat_to_tan_norm(joint_rot_flat)
    joint_rot_obs = torch.reshape(joint_rot_obs_flat, [joint_rot.shape[0], joint_rot.shape[1], joint_rot.shape[2] * joint_rot_obs_flat.shape[-1]])
    
    body_pos = torch.reshape(body_pos, [body_pos.shape[0], body_pos.shape[1], body_pos.shape[2] * body_pos.shape[3]])
    
    obs = [root_pos_obs, root_rot_obs, joint_rot_obs, body_pos]
    obs = torch.cat(obs, dim=-1)

    return obs

@torch.jit.script
def compute_disc_vel_obs(root_rot, root_vel, root_ang_vel, dof_vel, global_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor

    if not global_obs:
        heading_inv_rot = torch_util.calc_heading_quat_inv(root_rot)
        root_vel_obs = torch_util.quat_rotate(heading_inv_rot, root_vel)
        root_ang_vel_obs = torch_util.quat_rotate(heading_inv_rot, root_ang_vel)
    else:
        root_vel_obs = root_vel
        root_ang_vel_obs = root_ang_vel

    obs = [root_vel_obs, root_ang_vel_obs, dof_vel]
    obs = torch.cat(obs, dim=-1)

    return obs

@torch.jit.script
def compute_disc_obs(root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, body_pos, global_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool) -> Tensor

    pos_obs = compute_pos_obs(root_pos=root_pos, 
                            root_rot=root_rot, 
                            joint_rot=joint_rot, 
                            body_pos=body_pos,
                            global_obs=global_obs)

    vel_obs = compute_disc_vel_obs(root_rot=root_rot,
                                  root_vel=root_vel, 
                                  root_ang_vel=root_ang_vel, 
                                  dof_vel=dof_vel,
                                  global_obs=global_obs)

    disc_obs = torch.cat([pos_obs, vel_obs], dim=-1)
    disc_obs = torch.reshape(disc_obs, [disc_obs.shape[0], -1])

    return disc_obs