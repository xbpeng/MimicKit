import enum
import gymnasium.spaces as spaces
import numpy as np
import os
import torch

import anim.motion_lib as motion_lib
import engines.engine as engine
import envs.sim_env as sim_env
import envs.base_env as base_env
import util.camera as camera
from util.logger import Logger
import util.torch_util as torch_util

import engines.engine as engine

class CharEnv(sim_env.SimEnv):
    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        self._global_obs = env_config["global_obs"]
        self._root_height_obs = env_config.get("root_height_obs", True)
        self._zero_center_action = env_config.get("zero_center_action", False)
        
        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize, 
                         record_video=record_video)
        
        char_id = self._get_char_id()
        self._print_char_prop(0, char_id)
        self._validate_envs()
        return

    def _parse_init_pose(self, init_pose, device):
        dof_size = self._kin_char_model.get_dof_size()

        if (init_pose is None):
            init_pose = torch.zeros(6 + dof_size, dtype=torch.float32, device=device)
        else:
            init_pose = torch.tensor(init_pose, device=device)

            if (init_pose.shape[-1] == 3):
                pad_pose = torch.zeros(3 + dof_size, dtype=torch.float32, device=device)
                init_pose = torch.cat([init_pose, pad_pose], dim=-1)
            
        init_root_pos, init_root_rot, init_dof_pos = motion_lib.extract_pose_data(init_pose)
        assert(init_dof_pos.shape[-1] == dof_size)

        self._init_root_pos = init_root_pos
        self._init_root_rot = torch_util.exp_map_to_quat(init_root_rot)
        self._init_dof_pos = init_dof_pos
        return

    def _build_envs(self, env_config, num_envs):
        char_file = env_config["char_file"]
        self._build_kin_char_model(char_file)
        
        init_pose = env_config.get("init_pose", None)
        self._parse_init_pose(init_pose, self._device)
        
        self._char_ids = []
        
        for e in range(num_envs):
            Logger.print("Building {:d}/{:d} envs".format(e + 1, num_envs), end='\r')
            env_id = self._engine.create_env()
            assert(env_id == e)
            self._build_env(env_id, env_config)

        Logger.print("\n")
        return
    
    def _build_env(self, env_id, env_config):
        char_col = self._get_char_color()
        char_id = self._build_character(env_id, env_config, color=char_col)

        if (env_id == 0):
            self._char_ids.append(char_id)
        else:
            char_id0 = self._char_ids[0]
            assert(char_id0 == char_id)
        
        return 
    
    def _build_character(self, env_id, env_config, color=None):
        char_file = env_config["char_file"]
        char_id = self._engine.create_obj(env_id=env_id, 
                                          obj_type=engine.ObjType.articulated,
                                          asset_file=char_file, 
                                          name="character",
                                          start_pos=self._init_root_pos.cpu().numpy(),
                                          start_rot=self._init_root_rot.cpu().numpy(),
                                          color=color)
        return char_id
    
    def _build_kin_char_model(self, char_file):
        _, file_ext = os.path.splitext(char_file)
        if (file_ext == ".xml"):
            import anim.mjcf_char_model as mjcf_char_model
            char_model = mjcf_char_model.MJCFCharModel(self._device)
        elif (file_ext == ".urdf"):
            import anim.urdf_char_model as urdf_char_model
            char_model = urdf_char_model.URDFCharModel(self._device)
        elif (file_ext == ".usd"):
            import anim.usd_char_model as usd_char_model
            char_model = usd_char_model.USDCharModel(self._device)
        else:
            print("Unsupported character file format: {:s}".format(file_ext))
            assert(False)

        self._kin_char_model = char_model
        self._kin_char_model.load(char_file)
        return
    
    def _build_sim_tensors(self, env_config):
        super()._build_sim_tensors(env_config)
        
        self._action_bound_low = torch.tensor(self._action_space.low, device=self._device)
        self._action_bound_high = torch.tensor(self._action_space.high, device=self._device)
        
        key_bodies = env_config.get("key_bodies", [])
        self._key_body_ids = self._build_body_ids_tensor(key_bodies)
        return
    
    def _build_action_space(self):
        control_mode = self._engine.get_control_mode()

        if (control_mode == engine.ControlMode.none):
            low, high = self._build_action_bounds_none()

        elif (control_mode == engine.ControlMode.vel):
            low, high = self._build_action_bounds_vel()

        elif (control_mode == engine.ControlMode.torque):
            char_id = self._get_char_id()
            torque_lim = self._engine.get_obj_torque_limits(0, char_id)
            low, high = self._build_action_bounds_torque(torque_lim)

        elif (control_mode == engine.ControlMode.pos
              or control_mode == engine.ControlMode.pd_explicit):
            char_id = self._get_char_id()
            dof_low, dof_high = self._engine.get_obj_dof_limits(0, char_id)
            low, high = self._build_action_bounds_pos(dof_low, dof_high)

        else:
            assert(False), "Unsupported control mode: {}".format(control_mode)
        
        # check to make sure that pd_explicit is only used for 1D joints
        if (control_mode == engine.ControlMode.pd_explicit):
            num_joints = self._kin_char_model.get_num_joints()
            for j in range(1, num_joints):
                j_dim = self._kin_char_model.get_joint_dof_dim(j)
                assert(j_dim <= 1), "pd_explicit only supports 1D joints"

        action_space = spaces.Box(low=low, high=high)
        return action_space
    
    def _build_action_bounds_none(self):
        char_id = self._get_char_id()
        dof_pos = self._engine.get_dof_pos(char_id)
        action_size = int(dof_pos.shape[-1])
        low = -np.ones([action_size], dtype=np.float32)
        high = np.ones([action_size], dtype=np.float32)
        return low, high

    def _build_action_bounds_pos(self, dof_low, dof_high):
        low = np.zeros(dof_high.shape, dtype=np.float32)
        high = np.zeros(dof_high.shape, dtype=np.float32)

        num_joints = self._kin_char_model.get_num_joints()
        for j in range(1, num_joints):
            curr_joint = self._kin_char_model.get_joint(j)
            j_dof_dim = curr_joint.get_dof_dim()

            if (j_dof_dim > 0):
                if (j_dof_dim == 3): # 3D spherical j
                    # spherical joints are modeled as exponential maps
                    # so the bounds are computed a bit differently from revolute joints
                    j_low = curr_joint.get_joint_dof(dof_low)
                    j_high = curr_joint.get_joint_dof(dof_high)
                    j_low = np.max(np.abs(j_low))
                    j_high = np.max(np.abs(j_high))
                    curr_scale = max([j_low, j_high])
                    curr_scale = 1.2 * curr_scale

                    curr_low = -curr_scale
                    curr_high = curr_scale
                else:
                    j_low = curr_joint.get_joint_dof(dof_low)
                    j_high = curr_joint.get_joint_dof(dof_high)

                    if (self._zero_center_action):
                        curr_mid = np.zeros_like(j_high)
                    else:
                        curr_mid = 0.5 * (j_high + j_low)

                    diff_high = np.abs(j_high - curr_mid)
                    diff_low = np.abs(j_low - curr_mid)
                    curr_scale = np.maximum(diff_high, diff_low)
                    curr_scale *= 1.4

                    curr_low = curr_mid - curr_scale
                    curr_high = curr_mid + curr_scale

                curr_joint.set_joint_dof(curr_low, low)
                curr_joint.set_joint_dof(curr_high, high)

        return low, high

    def _build_action_bounds_vel(self):
        char_id = self._get_char_id()
        dof_pos = self._engine.get_dof_pos(char_id)
        action_size = int(dof_pos.shape[-1])
        low = -2.0 * np.pi * np.ones([action_size], dtype=np.float32)
        high = 2.0 * np.pi * np.ones([action_size], dtype=np.float32)
        return low, high

    def _build_action_bounds_torque(self, torque_lim):
        char_id = self._get_char_id()
        dof_pos = self._engine.get_dof_pos(char_id)
        assert(dof_pos.shape[-1] == len(torque_lim))
        low = -np.array(torque_lim, dtype=np.float32)
        high = np.array(torque_lim, dtype=np.float32)
        return low, high
    
    def _print_char_prop(self, env_id, obj_id):
        num_dofs = self._engine.get_obj_num_dofs(obj_id)
        total_mass = self._engine.calc_obj_mass(env_id, obj_id)
        char_info = "Char {:d} properties\n\tDoFs: {:d}\n\tMass: {:.3f} kg\n".format(obj_id, num_dofs, total_mass)
        Logger.print(char_info)
        return
    
    def _validate_envs(self):
        # checks to make sure the kinematic model is consistent with the simulation model
        char_id = self._get_char_id()
        sim_body_names = self._engine.get_obj_body_names(char_id)
        kin_body_names = self._kin_char_model.get_body_names()

        for sim_name, kin_name in zip(sim_body_names, kin_body_names):
            assert(sim_name == kin_name)
        return
    
    def _get_char_id(self):
        return self._char_ids[0]
    
    def _update_reward(self):
        char_id = self._get_char_id()
        char_root_pos = self._engine.get_root_pos(char_id)
        self._reward_buf[:] = compute_reward(char_root_pos)
        return

    def _update_done(self):
        self._done_buf[:] = compute_done(self._done_buf, self._time_buf, 
                                         self._episode_length)
        return
    
    def _compute_obs(self, env_ids=None):
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

        joint_rot = self._kin_char_model.dof_to_rot(dof_pos)

        if (self._has_key_bodies()):
            key_pos = body_pos[..., self._key_body_ids, :]
        else:
            key_pos = torch.zeros([0], device=self._device)

        obs = compute_char_obs(root_pos=root_pos,
                               root_rot=root_rot, 
                               root_vel=root_vel,
                               root_ang_vel=root_ang_vel,
                               joint_rot=joint_rot,
                               dof_vel=dof_vel,
                               key_pos=key_pos,
                               global_obs=self._global_obs,
                               root_height_obs=self._root_height_obs)
        return obs
    
    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)

        if (len(env_ids) > 0):
            self._reset_char(env_ids)
            self._reset_char_rigid_body_state(env_ids)
        return

    def _reset_char(self, env_ids):
        char_id = self._get_char_id()

        self._engine.set_root_pos(env_ids, char_id, self._init_root_pos)
        self._engine.set_root_rot(env_ids, char_id, self._init_root_rot)
        self._engine.set_root_vel(env_ids, char_id, 0.0)
        self._engine.set_root_ang_vel(env_ids, char_id, 0.0)
        
        self._engine.set_dof_pos(env_ids, char_id, self._init_dof_pos)
        self._engine.set_dof_vel(env_ids, char_id, 0.0)
        
        self._engine.set_body_vel(env_ids, char_id, 0.0)
        self._engine.set_body_ang_vel(env_ids, char_id, 0.0)
        return

    def _reset_char_rigid_body_state(self, env_ids):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)[env_ids]
        root_rot = self._engine.get_root_rot(char_id)[env_ids]
        dof_pos = self._engine.get_dof_pos(char_id)[env_ids]

        joint_rot = self._kin_char_model.dof_to_rot(dof_pos)
        body_pos, body_rot = self._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)

        self._engine.set_body_pos(env_ids, char_id, body_pos)
        self._engine.set_body_rot(env_ids, char_id, body_rot)
        return

    def _apply_action(self, actions):
        char_id = self._get_char_id()
        clip_action = torch.minimum(torch.maximum(actions, self._action_bound_low), self._action_bound_high)
        self._engine.set_cmd(char_id, clip_action)
        return
    
    def _build_body_ids_tensor(self, body_names):
        char_id = self._get_char_id()
        body_ids = []

        for body_name in body_names:
            body_id = self._engine.find_obj_body_id(char_id, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = torch.tensor(body_ids, device=self._device, dtype=torch.long)
        return body_ids
    
    def _has_key_bodies(self):
        return len(self._key_body_ids) > 0
    
    def _build_camera(self, env_config):
        env_id = 0
        char_id = self._get_char_id()
        char_root_pos = self._engine.get_root_pos(char_id)
        char_pos = char_root_pos[env_id].cpu().numpy()
            
        cam_pos = np.array([char_pos[0], char_pos[1] - 5.0, 3.0])
        cam_target = np.array([char_pos[0], char_pos[1], 1.0])

        cam_mode = camera.CameraMode[env_config["camera_mode"]]
        self._camera = camera.Camera(mode=cam_mode,
                                     engine=self._engine,
                                     pos=cam_pos,
                                     target=cam_target,
                                     track_env_id=env_id,
                                     track_obj_id=char_id)
        return
    
    def _get_char_color(self):
        engine_name = self._engine.get_name()
        if (engine_name == "isaac_lab"):
            col = np.array([0.2, 0.25, 0.7])
        elif (engine_name == "newton"):
            col = np.array([0.35, 0.45, 0.7])
        else:
            col = np.array([0.5, 0.65, 0.95])
        return col


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def convert_to_local_body_pos(root_rot, body_pos):
    # type: (Tensor, Tensor) -> Tensor
    
    heading_inv_rot = torch_util.calc_heading_quat_inv(root_rot)
    heading_rot_expand = heading_inv_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot_expand = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                            heading_rot_expand.shape[2])
    flat_body_pos = body_pos.reshape(body_pos.shape[0] * body_pos.shape[1], body_pos.shape[2])
    flat_local_body_pos = torch_util.quat_rotate(flat_heading_rot_expand, flat_body_pos)
    local_body_pos = flat_local_body_pos.reshape(body_pos.shape[0], body_pos.shape[1], body_pos.shape[2])

    return local_body_pos

@torch.jit.script
def convert_to_local_root_body_pos(root_rot, body_pos):
    # type: (Tensor, Tensor) -> Tensor
    
    root_inv_rot = torch_util.quat_conjugate(root_rot)
    root_rot_expand = root_inv_rot.unsqueeze(-2)
    root_rot_expand = root_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_root_rot_expand = root_rot_expand.reshape(root_rot_expand.shape[0] * root_rot_expand.shape[1], 
                                                   root_rot_expand.shape[2])
    flat_body_pos = body_pos.reshape(body_pos.shape[0] * body_pos.shape[1], body_pos.shape[2])
    flat_local_body_pos = torch_util.quat_rotate(flat_root_rot_expand, flat_body_pos)
    local_body_pos = flat_local_body_pos.reshape(body_pos.shape[0], body_pos.shape[1], body_pos.shape[2])

    return local_body_pos

@torch.jit.script
def compute_char_obs(root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos, global_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    heading_rot = torch_util.calc_heading_quat_inv(root_rot)
    
    if (global_obs):
        root_rot_obs = torch_util.quat_to_tan_norm(root_rot)
        root_vel_obs = root_vel
        root_ang_vel_obs = root_ang_vel
    else:
        local_root_rot = torch_util.quat_mul(heading_rot, root_rot)
        root_rot_obs = torch_util.quat_to_tan_norm(local_root_rot)
        root_vel_obs = torch_util.quat_rotate(heading_rot, root_vel)
        root_ang_vel_obs = torch_util.quat_rotate(heading_rot, root_ang_vel)

    joint_rot_flat = torch.reshape(joint_rot, [joint_rot.shape[0] * joint_rot.shape[1], joint_rot.shape[2]])
    joint_rot_obs_flat = torch_util.quat_to_tan_norm(joint_rot_flat)
    joint_rot_obs = torch.reshape(joint_rot_obs_flat, [joint_rot.shape[0], joint_rot.shape[1] * joint_rot_obs_flat.shape[-1]])

    obs = [root_rot_obs, root_vel_obs, root_ang_vel_obs, joint_rot_obs, dof_vel]

    if (len(key_pos) > 0):
        root_pos_expand = root_pos.unsqueeze(-2)
        key_pos = key_pos - root_pos_expand
        if (not global_obs):
            key_pos = convert_to_local_body_pos(root_rot, key_pos)

        key_pos_flat = torch.reshape(key_pos, [key_pos.shape[0], key_pos.shape[1] * key_pos.shape[2]])
        obs = obs + [key_pos_flat]

    if (root_height_obs):
        root_h = root_pos[:, 2:3]
        obs = [root_h] + obs
    
    obs = torch.cat(obs, dim=-1)
    return obs

@torch.jit.script
def compute_reward(root_pos):
    # type: (Tensor) -> Tensor
    r = torch.ones_like(root_pos[..., 0])
    return r

@torch.jit.script
def compute_done(done_buf, time, ep_len):
    # type: (Tensor, Tensor, float) -> Tensor
    timeout = time >= ep_len
    done = torch.full_like(done_buf, base_env.DoneFlags.NULL.value)
    done[timeout] = base_env.DoneFlags.TIME.value
    return done