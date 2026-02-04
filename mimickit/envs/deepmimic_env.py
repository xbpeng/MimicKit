import numpy as np
import torch

import anim.motion as motion
import anim.motion_lib as motion_lib
import envs.base_env as base_env
import envs.char_env as char_env
import engines.engine as engine
import util.stats_tracker as stats_tracker
import util.torch_util as torch_util

class DeepMimicEnv(char_env.CharEnv):
    def __init__(self, env_config, engine_config, num_envs, device, visualize):
        self._enable_early_termination = env_config["enable_early_termination"]
        self._num_phase_encoding = env_config.get("num_phase_encoding", 0)

        self._pose_termination = env_config.get("pose_termination", False)
        self._pose_termination_dist = env_config.get("pose_termination_dist", 1.0)
        self._enable_phase_obs = env_config.get("enable_phase_obs", True)
        self._enable_tar_obs = env_config.get("enable_tar_obs", False)
        self._tar_obs_steps = env_config.get("tar_obs_steps", [1])
        self._tar_obs_steps = torch.tensor(self._tar_obs_steps, device=device, dtype=torch.int)
        self._rand_reset = env_config.get("rand_reset", True)
        
        self._ref_char_offset = torch.tensor(env_config["ref_char_offset"], device=device, dtype=torch.float)
        self._log_tracking_error = env_config.get("log_tracking_error", False)
        
        self._reward_pose_w = env_config.get("reward_pose_w")
        self._reward_vel_w = env_config.get("reward_vel_w")
        self._reward_root_pose_w = env_config.get("reward_root_pose_w")
        self._reward_root_vel_w = env_config.get("reward_root_vel_w")
        self._reward_key_pos_w = env_config.get("reward_key_pos_w")

        self._reward_pose_scale = env_config.get("reward_pose_scale")
        self._reward_vel_scale = env_config.get("reward_vel_scale")
        self._reward_root_pose_scale = env_config.get("reward_root_pose_scale")
        self._reward_root_vel_scale = env_config.get("reward_root_vel_scale")
        self._reward_key_pos_scale = env_config.get("reward_key_pos_scale")
        
        self._visualize_ref_char = env_config.get("visualize_ref_char", True)
        
        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize)
        return
    
    def get_reward_succ(self):
        # setting the done flag flat to fail at the end of the motion avoids the
        # local minimal of a character just standing still until the end of the motion
        return 0.0
    
    def get_reward_fail(self):
        return 0.0
    
    def set_mode(self, mode):
        super().set_mode(mode)

        if (self._mode == base_env.EnvMode.TEST):
            if (self._log_tracking_error):
                self._error_tracker.reset()
        return

    def _build_sim_tensors(self, env_config):
        super()._build_sim_tensors(env_config)
        
        num_envs = self.get_num_envs()
        self._motion_ids = torch.zeros(num_envs, device=self._device, dtype=torch.int64)
        self._motion_time_offsets = torch.zeros(num_envs, device=self._device, dtype=torch.float32)
        
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        root_vel = self._engine.get_root_vel(char_id)
        root_ang_vel = self._engine.get_root_ang_vel(char_id)
        body_pos = self._engine.get_body_pos(char_id)
        body_rot = self._engine.get_body_rot(char_id)
        dof_pos = self._engine.get_dof_pos(char_id)
        dof_vel = self._engine.get_dof_vel(char_id)

        self._ref_root_pos = torch.zeros_like(root_pos)
        self._ref_root_rot = torch.zeros_like(root_rot)
        self._ref_root_vel = torch.zeros_like(root_vel)
        self._ref_root_ang_vel = torch.zeros_like(root_ang_vel)
        self._ref_body_pos = torch.zeros_like(body_pos)
        self._ref_body_rot = torch.zeros_like(body_rot)
        self._ref_joint_rot = torch.zeros_like(body_rot[..., 1:, :])
        self._ref_dof_pos = torch.zeros_like(dof_pos) 
        self._ref_dof_vel = torch.zeros_like(dof_vel)
        
        contact_bodies = env_config.get("contact_bodies", [])
        self._contact_body_ids = self._build_body_ids_tensor(contact_bodies)

        joint_err_w = env_config.get("joint_err_w", None)
        self._parse_joint_err_weights(joint_err_w)
        return

    def _load_motions(self, motion_file):
        self._motion_lib = motion_lib.MotionLib(motion_file=motion_file, 
                                                kin_char_model=self._kin_char_model,
                                                device=self._device)
        return
    
    def _parse_joint_err_weights(self, joint_err_w):
        num_joints = self._kin_char_model.get_num_joints()

        if (joint_err_w is None):
            self._joint_err_w = torch.ones(num_joints - 1, device=self._device, dtype=torch.float32)
        else:
            self._joint_err_w = torch.tensor(joint_err_w, device=self._device, dtype=torch.float32)

        assert(self._joint_err_w.shape[-1] == num_joints - 1)
        
        dof_size = self._kin_char_model.get_dof_size()
        self._dof_err_w = torch.zeros(dof_size, device=self._device, dtype=torch.float32)

        for j in range(1, num_joints):
            dof_dim = self._kin_char_model.get_joint_dof_dim(j)
            if (dof_dim > 0):
                curr_w = self._joint_err_w[j - 1]
                dof_idx = self._kin_char_model.get_joint_dof_idx(j)
                self._dof_err_w[dof_idx:dof_idx + dof_dim] = curr_w
        return
    
    def _enable_ref_char(self):
        return self._visualize and self._visualize_ref_char

    def _get_ref_char_color(self):
        engine_name = self._engine.get_name()
        if (engine_name == "isaac_lab"):
            col = np.array([0.25, 0.4, 0.1])
        elif (engine_name == "newton"):
            col = np.array([0.3, 0.5, 0.1])
        else:
            col = np.array([0.5, 0.9, 0.1])
        return col

    def _reset_char(self, env_ids):
        self._reset_ref_motion(env_ids)
        self._ref_state_init(env_ids)

        if (self._enable_ref_char()):
            self._reset_ref_char(env_ids)
        return

    def _reset_ref_char(self, env_ids):
        ref_char_id = self._get_ref_char_id()

        root_pos = self._ref_root_pos[env_ids] + self._ref_char_offset
        self._engine.set_root_pos(env_ids, ref_char_id, root_pos)
        self._engine.set_root_rot(env_ids, ref_char_id, self._ref_root_rot[env_ids])
        self._engine.set_root_vel(env_ids, ref_char_id, self._ref_root_vel[env_ids])
        self._engine.set_root_ang_vel(env_ids, ref_char_id, self._ref_root_ang_vel[env_ids])
        
        self._engine.set_dof_pos(env_ids, ref_char_id, self._ref_dof_pos[env_ids])
        self._engine.set_dof_vel(env_ids, ref_char_id, self._ref_dof_vel[env_ids])
        
        self._engine.set_body_vel(env_ids, ref_char_id, 0.0)
        self._engine.set_body_ang_vel(env_ids, ref_char_id, 0.0)
        return

    def _reset_ref_motion(self, env_ids):
        n = len(env_ids)
        motion_ids, motion_times = self._sample_motion_times(n)
        self._motion_ids[env_ids] = motion_ids
        self._motion_time_offsets[env_ids] = motion_times

        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = self._motion_lib.calc_motion_frame(motion_ids, motion_times)

        self._ref_root_pos[env_ids] = root_pos
        self._ref_root_rot[env_ids] = root_rot
        self._ref_root_vel[env_ids] = root_vel
        self._ref_root_ang_vel[env_ids] = root_ang_vel
        self._ref_joint_rot[env_ids] = joint_rot
        self._ref_dof_vel[env_ids] = dof_vel
        
        ref_body_pos, ref_body_rot = self._kin_char_model.forward_kinematics(self._ref_root_pos, self._ref_root_rot,
                                                                             self._ref_joint_rot)
        self._ref_body_pos[:] = ref_body_pos
        self._ref_body_rot[:] = ref_body_rot

        dof_pos = self._motion_lib.joint_rot_to_dof(joint_rot)
        self._ref_dof_pos[env_ids] = dof_pos
        return

    def _get_ref_char_id(self):
        return self._ref_char_ids[0]

    def _ref_state_init(self, env_ids):
        char_id = self._get_char_id()
        
        self._engine.set_root_pos(env_ids, char_id, self._ref_root_pos[env_ids])
        self._engine.set_root_rot(env_ids, char_id, self._ref_root_rot[env_ids])
        self._engine.set_root_vel(env_ids, char_id, self._ref_root_vel[env_ids])
        self._engine.set_root_ang_vel(env_ids, char_id, self._ref_root_ang_vel[env_ids])
        
        self._engine.set_dof_pos(env_ids, char_id, self._ref_dof_pos[env_ids])
        self._engine.set_dof_vel(env_ids, char_id, self._ref_dof_vel[env_ids])
        
        self._engine.set_body_vel(env_ids, char_id, 0.0)
        self._engine.set_body_ang_vel(env_ids, char_id, 0.0)
        return

    def _get_motion_times(self, env_ids=None):
        if (env_ids is None):
            motion_times = self._time_buf + self._motion_time_offsets
        else:
            motion_times = self._time_buf[env_ids] + self._motion_time_offsets[env_ids]
        return motion_times

    def _update_misc(self):
        super()._update_misc()
        self._update_ref_motion()

        if (self._enable_ref_char()):
            self._update_ref_char()
        return
    
    def _update_ref_motion(self):
        motion_ids = self._motion_ids
        motion_times = self._get_motion_times()
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        
        self._ref_root_pos[:] = root_pos
        self._ref_root_rot[:] = root_rot
        self._ref_root_vel[:] = root_vel
        self._ref_root_ang_vel[:] = root_ang_vel
        self._ref_joint_rot[:] = joint_rot
        self._ref_dof_vel[:] = dof_vel

        ref_body_pos, ref_body_rot = self._kin_char_model.forward_kinematics(self._ref_root_pos, self._ref_root_rot,
                                                                             self._ref_joint_rot)
        self._ref_body_pos[:] = ref_body_pos
        self._ref_body_rot[:] = ref_body_rot

        if (self._enable_ref_char()):
            dof_pos = self._motion_lib.joint_rot_to_dof(joint_rot)
            self._ref_dof_pos[:] = dof_pos
        return

    def _update_ref_char(self):
        ref_char_id = self._get_ref_char_id()

        root_pos = self._ref_root_pos + self._ref_char_offset
        body_pos = self._ref_body_pos + self._ref_char_offset

        self._engine.set_root_pos(None, ref_char_id, root_pos)
        self._engine.set_root_rot(None, ref_char_id, self._ref_root_rot)
        self._engine.set_root_vel(None, ref_char_id, 0.0)
        self._engine.set_root_ang_vel(None, ref_char_id, 0.0)
        
        self._engine.set_dof_pos(None, ref_char_id, self._ref_dof_pos)
        self._engine.set_dof_vel(None, ref_char_id, 0.0)

        self._engine.set_body_pos(None, ref_char_id, body_pos)
        self._engine.set_body_rot(None, ref_char_id, self._ref_body_rot)
        self._engine.set_body_vel(None, ref_char_id, 0.0)
        self._engine.set_body_ang_vel(None, ref_char_id, 0.0)
        return
    
    def _track_global_root(self):
        return self._enable_tar_obs and self._global_obs

    def _sample_motion_times(self, n):
        motion_ids = self._motion_lib.sample_motions(n)

        if (self._rand_reset):
            motion_times = self._motion_lib.sample_time(motion_ids)
        else:
            motion_times = torch.zeros(n, dtype=torch.float, device=self._device)

        return motion_ids, motion_times

    def _build_data_buffers(self):
        super()._build_data_buffers()

        if (self._log_tracking_error):
            num_track_errors = 7
            self._error_tracker = stats_tracker.StatsTracker(num_track_errors, device=self._device)
        return
    
    def _build_envs(self, env_config, num_envs):
        self._ref_char_ids = []

        super()._build_envs(env_config, num_envs)

        motion_file = env_config["motion_file"]
        self._load_motions(motion_file)
        return
    
    def _build_env(self, env_id, config):
        super()._build_env(env_id, config)

        if (self._enable_ref_char()):
            ref_char_col = self._get_ref_char_color()
            ref_char_id = self._build_ref_character(env_id, config, color=ref_char_col)
            self._ref_char_ids.append(ref_char_id)
            
            if (env_id == 0):
                self._ref_char_ids.append(ref_char_id)
            else:
                ref_char_id0 = self._ref_char_ids[0]
                assert(ref_char_id0 == ref_char_id)
        return 
    
    def _build_ref_character(self, env_id, env_config, color):
        char_file = env_config["char_file"]
        char_id = self._engine.create_obj(env_id=env_id, 
                                          obj_type=engine.ObjType.articulated,
                                          asset_file=char_file, 
                                          name="ref_character",
                                          is_visual=True,
                                          enable_self_collisions=False,
                                          disable_motors=True,
                                          color=color)
        return char_id

    def _compute_obs(self, env_ids=None):
        motion_ids = self._motion_ids
        motion_times = self._get_motion_times(env_ids)
        
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        root_vel = self._engine.get_root_vel(char_id)
        root_ang_vel = self._engine.get_root_ang_vel(char_id)
        dof_pos = self._engine.get_dof_pos(char_id)
        dof_vel = self._engine.get_dof_vel(char_id)
        body_pos = self._engine.get_body_pos(char_id)

        if (env_ids is not None):
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]
            root_vel = root_vel[env_ids]
            root_ang_vel = root_ang_vel[env_ids]
            dof_pos = dof_pos[env_ids]
            dof_vel = dof_vel[env_ids]
            body_pos = body_pos[env_ids]

            motion_ids = motion_ids[env_ids]

        # Validate dof_pos before passing to dof_to_rot
        if not isinstance(dof_pos, torch.Tensor):
            raise TypeError(
                f"_compute_obs: dof_pos expected torch.Tensor, got {type(dof_pos).__name__}. "
                f"char_id={char_id}, env_ids={env_ids}"
            )

        joint_rot = self._kin_char_model.dof_to_rot(dof_pos)
        
        if (self._enable_phase_obs):
            motion_phase = self._motion_lib.calc_motion_phase(motion_ids, motion_times)
        else:
            motion_phase = torch.zeros([0], device=self._device)

        if (self._has_key_bodies()):
            key_pos = body_pos[..., self._key_body_ids, :]
        else:
            key_pos = torch.zeros([0], device=self._device)

        if (self._enable_tar_obs):
            tar_root_pos, tar_root_rot, tar_joint_rot = self._fetch_tar_obs_data(motion_ids, motion_times)
            tar_root_pos_flat = torch.reshape(tar_root_pos, [tar_root_pos.shape[0] * tar_root_pos.shape[1], 
                                                             tar_root_pos.shape[-1]])
            tar_root_rot_flat = torch.reshape(tar_root_rot, [tar_root_rot.shape[0] * tar_root_rot.shape[1], 
                                                             tar_root_rot.shape[-1]])
            tar_joint_rot_flat = torch.reshape(tar_joint_rot, [tar_joint_rot.shape[0] * tar_joint_rot.shape[1], 
                                                               tar_joint_rot.shape[-2], tar_joint_rot.shape[-1]])
            tar_body_pos_flat, _ = self._kin_char_model.forward_kinematics(tar_root_pos_flat, tar_root_rot_flat,
                                                                           tar_joint_rot_flat)
            tar_body_pos = torch.reshape(tar_body_pos_flat, [tar_root_pos.shape[0], tar_root_pos.shape[1], 
                                                             tar_body_pos_flat.shape[-2], tar_body_pos_flat.shape[-1]])

            if (self._has_key_bodies()):
                tar_key_pos = tar_body_pos[..., self._key_body_ids, :]
            else:
                tar_key_pos = torch.zeros([0], device=self._device)
        else:
            tar_root_pos = torch.zeros([0], device=self._device)
            tar_root_rot = tar_root_pos
            tar_joint_rot = tar_root_pos
            tar_key_pos = tar_root_pos

        obs = compute_deepmimic_obs(root_pos=root_pos, 
                                    root_rot=root_rot, 
                                    root_vel=root_vel, 
                                    root_ang_vel=root_ang_vel,
                                    joint_rot=joint_rot,
                                    dof_vel=dof_vel,
                                    key_pos=key_pos,
                                    global_obs=self._global_obs,
                                    root_height_obs=self._root_height_obs,
                                    phase=motion_phase,
                                    num_phase_encoding=self._num_phase_encoding,
                                    enable_phase_obs=self._enable_phase_obs,
                                    enable_tar_obs=self._enable_tar_obs,
                                    tar_root_pos=tar_root_pos,
                                    tar_root_rot=tar_root_rot,
                                    tar_joint_rot=tar_joint_rot,
                                    tar_key_pos=tar_key_pos)
        return obs
    
    def _update_reward(self):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        root_vel = self._engine.get_root_vel(char_id)
        root_ang_vel = self._engine.get_root_ang_vel(char_id)
        dof_pos = self._engine.get_dof_pos(char_id)
        dof_vel = self._engine.get_dof_vel(char_id)
        body_pos = self._engine.get_body_pos(char_id)

        # Validate dof_pos before passing to dof_to_rot
        if not isinstance(dof_pos, torch.Tensor):
            raise TypeError(
                f"_update_reward: dof_pos expected torch.Tensor, got {type(dof_pos).__name__}"
            )

        joint_rot = self._kin_char_model.dof_to_rot(dof_pos)
        if (self._has_key_bodies()):
            key_pos = body_pos[..., self._key_body_ids, :]
            ref_key_pos = self._ref_body_pos[..., self._key_body_ids, :]
        else:
            key_pos = torch.zeros([0], device=self._device)
            ref_key_pos = key_pos

        track_root_h = self._root_height_obs
        track_root = self._track_global_root()
        
        self._reward_buf[:] = compute_reward(root_pos=root_pos,
                                             root_rot=root_rot,
                                             root_vel=root_vel,
                                             root_ang_vel=root_ang_vel,
                                             joint_rot=joint_rot,
                                             dof_vel=dof_vel,
                                             key_pos=key_pos,
                                             
                                             tar_root_pos=self._ref_root_pos,
                                             tar_root_rot=self._ref_root_rot,
                                             tar_root_vel=self._ref_root_vel,
                                             tar_root_ang_vel=self._ref_root_ang_vel,
                                             tar_joint_rot=self._ref_joint_rot,
                                             tar_dof_vel=self._ref_dof_vel,
                                             tar_key_pos=ref_key_pos,
                                             
                                             joint_rot_err_w=self._joint_err_w,
                                             dof_err_w=self._dof_err_w,
                                             track_root_h=track_root_h,
                                             track_root=track_root,               
                                             
                                             pose_w=self._reward_pose_w,
                                             vel_w=self._reward_vel_w,
                                             root_pose_w=self._reward_root_pose_w,
                                             root_vel_w=self._reward_root_vel_w,
                                             key_pos_w=self._reward_key_pos_w,

                                             pose_scale=self._reward_pose_scale,
                                             vel_scale=self._reward_vel_scale,
                                             root_pose_scale=self._reward_root_pose_scale,
                                             root_vel_scale=self._reward_root_vel_scale,
                                             key_pos_scale=self._reward_key_pos_scale)
        return

    def _update_done(self):
        motion_times = self._get_motion_times()
        motion_len = self._motion_lib.get_motion_length(self._motion_ids)
        motion_loop_mode = self._motion_lib.get_motion_loop_mode(self._motion_ids)
        motion_len_term = motion_loop_mode != motion.LoopMode.WRAP.value

        track_root = self._track_global_root()
        
        char_id = self._get_char_id()
        root_rot = self._engine.get_root_rot(char_id)
        body_pos = self._engine.get_body_pos(char_id)
        ground_contact_forces = self._engine.get_ground_contact_forces(char_id)

        self._done_buf[:] = compute_done(done_buf=self._done_buf,
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

    def _update_info(self, env_ids=None):
        super()._update_info(env_ids)
        
        if (self._mode == base_env.EnvMode.TEST):
            if (self._log_tracking_error):
                self._record_tracking_error(env_ids)
        return
    
    def _record_tracking_error(self, env_ids=None):
        if (env_ids is None or len(env_ids) > 0):
            char_id = self._get_char_id()
            root_pos = self._engine.get_root_pos(char_id)
            root_rot = self._engine.get_root_rot(char_id)
            root_vel = self._engine.get_root_vel(char_id)
            root_ang_vel = self._engine.get_root_ang_vel(char_id)
            dof_pos = self._engine.get_dof_pos(char_id)
            dof_vel = self._engine.get_dof_vel(char_id)
            body_pos = self._engine.get_body_pos(char_id)
            body_rot = self._engine.get_body_rot(char_id)

            joint_rot = self._kin_char_model.dof_to_rot(dof_pos)

            ref_root_pos = self._ref_root_pos
            ref_root_rot = self._ref_root_rot
            ref_joint_rot = self._ref_joint_rot
            ref_root_vel = self._ref_root_vel
            ref_root_ang_vel = self._ref_root_ang_vel
            ref_dof_vel = self._ref_dof_vel

            if env_ids is not None:
                root_pos = root_pos[env_ids]
                root_rot = root_rot[env_ids]
                joint_rot = joint_rot[env_ids]
                root_vel = root_vel[env_ids]
                root_ang_vel = root_ang_vel[env_ids]
                dof_vel = dof_vel[env_ids]
                body_pos = body_pos[env_ids]
                body_rot = body_rot[env_ids]

                ref_root_pos = ref_root_pos[env_ids]
                ref_root_rot = ref_root_rot[env_ids]
                ref_joint_rot = ref_joint_rot[env_ids]
                ref_root_vel = ref_root_vel[env_ids]
                ref_root_ang_vel = ref_root_ang_vel[env_ids]
                ref_dof_vel = ref_dof_vel[env_ids]
            
            ref_body_pos, ref_body_rot = self._kin_char_model.forward_kinematics(ref_root_pos, ref_root_rot, ref_joint_rot)

            tracking_error = compute_tracking_error(root_pos=root_pos,
                                                    root_rot=root_rot,
                                                    body_rot=body_rot,
                                                    body_pos=body_pos,

                                                    tar_root_pos=ref_root_pos,
                                                    tar_root_rot=ref_root_rot,
                                                    tar_body_rot=ref_body_rot,
                                                    tar_body_pos=ref_body_pos,

                                                    root_vel=root_vel,
                                                    root_ang_vel=root_ang_vel,
                                                    dof_vel=dof_vel,
                                                    tar_dof_vel=ref_dof_vel,
                                                    tar_root_vel=ref_root_vel,
                                                    tar_root_ang_vel=ref_root_ang_vel)

            self._error_tracker.update(tracking_error)

            err_stats = self._error_tracker.get_mean()
            self._diagnostics["root_pos_err"] = err_stats[0]
            self._diagnostics["root_rot_err"] = err_stats[1]
            self._diagnostics["body_pos_err"] = err_stats[2]
            self._diagnostics["body_rot_err"] = err_stats[3]
            self._diagnostics["dof_vel_err"] = err_stats[4]
            self._diagnostics["root_vel_err"] = err_stats[5]
            self._diagnostics["root_ang_vel_err"] = err_stats[6]
        return
    
    def _fetch_tar_obs_data(self, motion_ids, motion_times):
        n = motion_ids.shape[0]
        num_steps = self._tar_obs_steps.shape[0]
        assert(num_steps > 0)
        
        motion_times = motion_times.unsqueeze(-1)
        time_steps = self._engine.get_timestep() * self._tar_obs_steps
        motion_times = motion_times + time_steps
        motion_ids_tiled = torch.broadcast_to(motion_ids.unsqueeze(-1), motion_times.shape)

        motion_ids_tiled = motion_ids_tiled.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = self._motion_lib.calc_motion_frame(motion_ids_tiled, motion_times)
        
        root_pos = root_pos.reshape([n, num_steps, root_pos.shape[-1]])
        root_rot = root_rot.reshape([n, num_steps, root_rot.shape[-1]])
        joint_rot = joint_rot.reshape([n, num_steps, joint_rot.shape[-2], joint_rot.shape[-1]])

        return root_pos, root_rot, joint_rot


@torch.jit.script
def compute_phase_obs(phase, num_phase_encoding):
    # type: (Tensor, int) -> Tensor
    phase_obs = phase.unsqueeze(-1)

    # positional embedding of phase
    if (num_phase_encoding > 0):
        pe_exp = torch.arange(num_phase_encoding, device=phase.device, dtype=phase.dtype)
        pe_scale = 2.0 * np.pi * torch.pow(2.0, pe_exp)
        pe_scale = pe_scale.unsqueeze(0)
        pe_val = phase.unsqueeze(-1) * pe_scale
        pe_sin = torch.sin(pe_val)
        pe_cos = torch.cos(pe_val)

        phase_obs = torch.cat((phase_obs, pe_sin, pe_cos), dim=-1)

    return phase_obs

@torch.jit.script
def convert_to_local(root_rot, root_vel, root_ang_vel, key_pos):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    heading_inv_rot = torch_util.calc_heading_quat_inv(root_rot)

    local_root_rot = torch_util.quat_mul(heading_inv_rot, root_rot)
    local_root_vel = torch_util.quat_rotate(heading_inv_rot, root_vel)
    local_root_ang_vel = torch_util.quat_rotate(heading_inv_rot, root_ang_vel)
    
    if (len(key_pos) > 0):
        heading_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_rot_expand = heading_rot_expand.repeat((1, key_pos.shape[1], 1))
        flat_heading_rot_expand = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                                heading_rot_expand.shape[2])
        flat_key_pos = key_pos.reshape(key_pos.shape[0] * key_pos.shape[1], key_pos.shape[2])
        flat_local_key_pos = torch_util.quat_rotate(flat_heading_rot_expand, flat_key_pos)
        local_key_pos = flat_local_key_pos.reshape(key_pos.shape[0], key_pos.shape[1], key_pos.shape[2])
    else:
        local_key_pos = key_pos

    return local_root_rot, local_root_vel, local_root_ang_vel, local_key_pos

@torch.jit.script
def compute_tar_obs(ref_root_pos, ref_root_rot, root_pos, root_rot, joint_rot, key_pos,
                    global_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    ref_root_pos = ref_root_pos.unsqueeze(-2)
    root_pos_obs = root_pos - ref_root_pos
    
    if (len(key_pos) > 0):
        key_pos = key_pos - root_pos.unsqueeze(-2)

    if (not global_obs):
        heading_inv_rot = torch_util.calc_heading_quat_inv(ref_root_rot)
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, root_pos.shape[1], 1))
        heading_inv_rot_flat = heading_inv_rot_expand.reshape((heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1], 
                                                               heading_inv_rot_expand.shape[2]))
        root_pos_obs_flat = torch.reshape(root_pos_obs, [root_pos_obs.shape[0] * root_pos_obs.shape[1], root_pos_obs.shape[2]])
        root_pos_obs_flat = torch_util.quat_rotate(heading_inv_rot_flat, root_pos_obs_flat)
        root_pos_obs = torch.reshape(root_pos_obs_flat, root_pos.shape)
        
        root_rot = torch_util.quat_mul(heading_inv_rot_expand, root_rot)

        if (len(key_pos) > 0):
            heading_inv_rot_expand = heading_inv_rot_expand.unsqueeze(-2)
            heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, 1, key_pos.shape[2], 1))
            heading_inv_rot_flat = heading_inv_rot_expand.reshape((heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1] * heading_inv_rot_expand.shape[2],
                                                                   heading_inv_rot_expand.shape[3]))
            key_pos_flat = key_pos.reshape((key_pos.shape[0] * key_pos.shape[1] * key_pos.shape[2],
                                            key_pos.shape[3]))
            key_pos_flat = torch_util.quat_rotate(heading_inv_rot_flat, key_pos_flat)
            key_pos = key_pos_flat.reshape(key_pos.shape)

    if (root_height_obs):
        root_pos_obs[..., 2] = root_pos[..., 2]
    else:
        root_pos_obs = root_pos_obs[..., :2]

    root_rot_flat = torch.reshape(root_rot, [root_rot.shape[0] * root_rot.shape[1], root_rot.shape[2]])
    root_rot_obs_flat = torch_util.quat_to_tan_norm(root_rot_flat)
    root_rot_obs = torch.reshape(root_rot_obs_flat, [root_rot.shape[0], root_rot.shape[1], root_rot_obs_flat.shape[-1]])

    joint_rot_flat = torch.reshape(joint_rot, [joint_rot.shape[0] * joint_rot.shape[1] * joint_rot.shape[2], joint_rot.shape[3]])
    joint_rot_obs_flat = torch_util.quat_to_tan_norm(joint_rot_flat)
    joint_rot_obs = torch.reshape(joint_rot_obs_flat, [joint_rot.shape[0], joint_rot.shape[1], joint_rot.shape[2] * joint_rot_obs_flat.shape[-1]])
    
    obs = [root_pos_obs, root_rot_obs, joint_rot_obs]
    if (len(key_pos) > 0):
        key_pos = torch.reshape(key_pos, [key_pos.shape[0], key_pos.shape[1], key_pos.shape[2] * key_pos.shape[3]])
        obs.append(key_pos)

    obs = torch.cat(obs, dim=-1)

    return obs

@torch.jit.script
def compute_deepmimic_obs(root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos, global_obs, root_height_obs, 
                          phase, num_phase_encoding, enable_phase_obs, 
                          enable_tar_obs, tar_root_pos, tar_root_rot, tar_joint_rot, tar_key_pos):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, Tensor, int, bool, bool, Tensor, Tensor, Tensor, Tensor) -> Tensor
    char_obs = char_env.compute_char_obs(root_pos=root_pos,
                                            root_rot=root_rot,
                                            root_vel=root_vel,
                                            root_ang_vel=root_ang_vel,
                                            joint_rot=joint_rot,
                                            dof_vel=dof_vel,
                                            key_pos=key_pos,
                                            global_obs=global_obs,
                                            root_height_obs=root_height_obs)
    obs = [char_obs]

    if (enable_phase_obs):
        phase_obs = compute_phase_obs(phase=phase, num_phase_encoding=num_phase_encoding)
        obs.append(phase_obs)

    if (enable_tar_obs):
        if (global_obs):
            ref_root_pos = root_pos
            ref_root_rot = root_rot
        else:
            ref_root_pos = tar_root_pos[..., 0, :]
            ref_root_rot = tar_root_rot[..., 0, :]

        tar_obs = compute_tar_obs(ref_root_pos=ref_root_pos,
                                  ref_root_rot=ref_root_rot,
                                  root_pos=tar_root_pos, 
                                  root_rot=tar_root_rot, 
                                  joint_rot=tar_joint_rot,
                                  key_pos=tar_key_pos,
                                  global_obs=global_obs,
                                  root_height_obs=root_height_obs)
        
        tar_obs = torch.reshape(tar_obs, [tar_obs.shape[0], tar_obs.shape[1] * tar_obs.shape[2]])
        obs.append(tar_obs)

    obs = torch.cat(obs, dim=-1)
    
    return obs

@torch.jit.script
def compute_done(done_buf, time, ep_len, root_rot, body_pos, tar_root_rot, tar_body_pos, 
                 ground_contact_force, contact_body_ids,
                 pose_termination, pose_termination_dist, 
                 global_obs, enable_early_termination,
                 motion_times, motion_len, motion_len_term,
                 track_root):
    # type: (Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, float, bool, bool, Tensor, Tensor, Tensor, bool) -> Tensor
    done = torch.full_like(done_buf, base_env.DoneFlags.NULL.value)
    
    timeout = time >= ep_len
    done[timeout] = base_env.DoneFlags.TIME.value
    
    motion_end = motion_times >= motion_len
    motion_end = torch.logical_and(motion_end, motion_len_term)
    done[motion_end] = base_env.DoneFlags.SUCC.value

    if (enable_early_termination):
        failed = torch.zeros(done.shape, device=done.device, dtype=torch.bool)

        if (contact_body_ids.shape[0] > 0):
            masked_contact_buf = ground_contact_force.detach().clone()
            masked_contact_buf[:, contact_body_ids, :] = 0
            fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)

            has_fallen = torch.any(fall_contact, dim=-1)
            failed = torch.logical_or(failed, has_fallen)

        if (pose_termination):
            root_pos = body_pos[..., 0:1, :]
            tar_root_pos = tar_body_pos[..., 0:1, :]

            if (not global_obs):
                body_pos = body_pos[..., 1:, :] - root_pos
                tar_body_pos = tar_body_pos[..., 1:, :] - tar_root_pos
                body_pos = char_env.convert_to_local_root_body_pos(root_rot, body_pos)
                tar_body_pos = char_env.convert_to_local_root_body_pos(tar_root_rot, tar_body_pos)

            elif (not track_root):
                body_pos = body_pos[..., 1:, :] - root_pos
                tar_body_pos = tar_body_pos[..., 1:, :] - tar_root_pos

            body_pos_diff = tar_body_pos - body_pos
            body_pos_dist = torch.sum(body_pos_diff * body_pos_diff, dim=-1)
            body_pos_dist = torch.max(body_pos_dist, dim=-1)[0]
            pose_fail = body_pos_dist > pose_termination_dist * pose_termination_dist

            if (track_root):
                root_pos_diff = tar_root_pos - root_pos
                root_pos_dist = torch.sum(root_pos_diff * root_pos_diff, dim=-1)
                root_pos_fail = root_pos_dist > pose_termination_dist * pose_termination_dist
                root_pos_fail = root_pos_fail.squeeze(-1)
                pose_fail = torch.logical_or(pose_fail, root_pos_fail)

            failed = torch.logical_or(failed, pose_fail)
            
        # only fail after first timestep
        not_first_step = (time > 0.0)
        failed = torch.logical_and(failed, not_first_step)
        done[failed] = base_env.DoneFlags.FAIL.value
    
    return done

@torch.jit.script
def compute_reward(root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos,
                   tar_root_pos, tar_root_rot, tar_root_vel, tar_root_ang_vel,
                   tar_joint_rot, tar_dof_vel, tar_key_pos,
                   joint_rot_err_w, dof_err_w, track_root_h, track_root,
                   pose_w, vel_w, root_pose_w, root_vel_w, key_pos_w,
                   pose_scale, vel_scale, root_pose_scale, root_vel_scale, key_pos_scale):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, float, float, float, float, float, float, float, float, float, float) -> Tensor
    pose_diff = torch_util.quat_diff_angle(joint_rot, tar_joint_rot)
    pose_err = torch.sum(joint_rot_err_w * pose_diff * pose_diff, dim=-1)

    vel_diff = tar_dof_vel - dof_vel
    vel_err = torch.sum(dof_err_w * vel_diff * vel_diff, dim=-1)

    root_pos_diff = tar_root_pos - root_pos

    if (not track_root):
        root_pos_diff[..., 0:2] = 0

    if (not track_root_h):
        root_pos_diff[..., 2] = 0

    root_pos_err = torch.sum(root_pos_diff * root_pos_diff, dim=-1)
    
    if (len(key_pos) > 0):
        key_pos = key_pos - root_pos.unsqueeze(-2)
        tar_key_pos = tar_key_pos - tar_root_pos.unsqueeze(-2)

    if (not track_root):
        root_rot, root_vel, root_ang_vel, key_pos = convert_to_local(root_rot, root_vel, root_ang_vel, key_pos)
        tar_root_rot, tar_root_vel, tar_root_ang_vel, tar_key_pos = convert_to_local(tar_root_rot, tar_root_vel, tar_root_ang_vel, tar_key_pos)
        
    root_rot_err = torch_util.quat_diff_angle(root_rot, tar_root_rot)
    root_rot_err *= root_rot_err

    root_vel_diff = tar_root_vel - root_vel
    root_vel_err = torch.sum(root_vel_diff * root_vel_diff, dim=-1)

    root_ang_vel_diff = tar_root_ang_vel - root_ang_vel
    root_ang_vel_err = torch.sum(root_ang_vel_diff * root_ang_vel_diff, dim=-1)

    if (len(key_pos) > 0):
        key_pos_diff = tar_key_pos - key_pos
        key_pos_err = torch.sum(key_pos_diff * key_pos_diff, dim=-1)
        key_pos_err = torch.sum(key_pos_err, dim=-1)
    else:
        key_pos_err = torch.zeros([0], device=key_pos.device)

    pose_r = torch.exp(-pose_scale * pose_err)
    vel_r = torch.exp(-vel_scale * vel_err)
    root_pose_r = torch.exp(-root_pose_scale * (root_pos_err + 0.1 * root_rot_err))
    root_vel_r = torch.exp(-root_vel_scale * (root_vel_err + 0.1 * root_ang_vel_err))
    key_pos_r = torch.exp(-key_pos_scale * key_pos_err)

    r = pose_w * pose_r \
        + vel_w * vel_r \
        + root_pose_w * root_pose_r \
        + root_vel_w * root_vel_r \
        + key_pos_w * key_pos_r

    return r

@torch.jit.script
def compute_tracking_error(root_pos, root_rot, body_rot, body_pos,
                            tar_root_pos, tar_root_rot,
                            tar_body_rot, tar_body_pos,
                            root_vel, root_ang_vel, dof_vel,
                            tar_root_vel, tar_root_ang_vel, tar_dof_vel):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    body_pos = body_pos - root_pos.unsqueeze(-2)
    tar_body_pos = tar_body_pos - tar_root_pos.unsqueeze(-2)

    root_pos_diff = tar_root_pos - root_pos
    root_pos_err = torch.linalg.vector_norm(root_pos_diff, dim=-1)

    body_rot_diff = torch_util.quat_diff_angle(body_rot, tar_body_rot)
    body_rot_err = torch.abs(body_rot_diff)
    body_rot_err = torch.mean(body_rot_err, dim=-1)

    body_pos_diff = tar_body_pos - body_pos
    body_pos_diff_l2 = torch.linalg.vector_norm(body_pos_diff, dim=-1)
    body_pos_err = torch.mean(body_pos_diff_l2, dim=-1)

    root_rot_diff = torch_util.quat_diff_angle(root_rot, tar_root_rot)
    root_rot_err = torch.abs(root_rot_diff)

    dof_vel_diff = tar_dof_vel - dof_vel
    dof_vel_err = torch.mean(torch.abs(dof_vel_diff), dim=-1)

    root_vel_diff = tar_root_vel - root_vel
    root_vel_err = torch.mean(torch.abs(root_vel_diff), dim=-1)

    root_ang_vel_diff = tar_root_ang_vel - root_ang_vel
    root_ang_vel_err = torch.mean(torch.abs(root_ang_vel_diff), dim=-1)

    tracking_error = torch.stack([root_pos_err, root_rot_err, body_pos_err, body_rot_err, dof_vel_err, root_vel_err, root_ang_vel_err], dim=-1)
    
    return tracking_error