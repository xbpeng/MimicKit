import isaacgym.gymapi as gymapi
import isaacgym.gymtorch as gymtorch

import numpy as np
import os
import re
import sys
import torch
import time

import engines.engine as engine
import util.torch_util as torch_util

def str_to_key_code(key_str):
    key_str = key_str.upper()

    if (key_str == "ESC"):
        key_name = "KEY_ESCAPE"
    elif (key_str == "RETURN"):
        key_name = "KEY_ENTER"
    elif (key_str == "DELETE"):
        key_name = "KEY_DEL"
    else:
        key_name = "KEY_" + key_str
    
    key_code = getattr(gymapi.KeyboardInput, key_name)
    return key_code

class AssetCfg():
    def __init__(self, asset_file, fix_root):
        self.asset_file = asset_file
        self.fix_root = fix_root
        return
    
    def __hash__(self):
        values = (self.asset_file, self.fix_root)
        asset_hash = hash(values)
        return asset_hash
    
    def __eq__(self, other):
        same = self.asset_file == other.asset_file
        same &= self.fix_root == other.fix_root
        return same


class IsaacGymEngine(engine.Engine):
    def __init__(self, config, num_envs, device, visualize):
        super().__init__()
        self._device = device
        self._num_envs = num_envs
        self._asset_cache = dict()
        
        self._gym = gymapi.acquire_gym()
        
        sim_freq = config.get("sim_freq", 60)
        control_freq = config.get("control_freq", 10)
        assert(sim_freq >= control_freq and sim_freq % control_freq == 0), \
            "Simulation frequency must be a multiple of the control frequency"

        self._timestep = 1.0 / control_freq
        self._sim_steps = int(sim_freq / control_freq)
        sim_timestep = 1.0 / sim_freq
        self._sim = self._create_simulator(sim_timestep, visualize)

        self._ground_contact_height = config.get("ground_contact_height", 0.3)
        self._env_spacing = config["env_spacing"]
        self._envs = []
        self._obj_types = []

        if ("control_mode" in config):
            self._control_mode = engine.ControlMode[config["control_mode"]]
        else:
            self._control_mode = engine.ControlMode.none

        self._obj_kp = [[] for i in range(num_envs)]
        self._obj_kd = [[] for i in range(num_envs)]
        self._obj_torque_lim = [[] for i in range(num_envs)]

        self._build_ground()

        if (visualize):
            self._build_viewer()
            self._keyboard_callbacks = dict()
            self._prev_frame_time = 0.0

        return

    def get_name(self):
        return "isaac_gym"
    
    def create_env(self):
        env_spacing = self._get_env_spacing() / 2.0
        num_envs = self.get_num_envs()
        num_env_per_row = int(np.sqrt(num_envs))

        lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
        upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        env_ptr = self._gym.create_env(self._sim, lower, upper, num_env_per_row)

        env_id = len(self._envs)
        assert(env_id < self.get_num_envs())
        
        self._envs.append(env_ptr)
        self._obj_types.append([])

        return env_id
    
    def initialize_sim(self):
        self._gym.prepare_sim(self._sim)
        self._build_sim_tensors()
        self._refresh_sim_tensors()
        return
    
    def step(self):
        self._update_reset_objs()
        self._apply_cmd()

        for i in range(self._sim_steps):
            self._pre_sim_step()
            self._sim_step()
            
        self._refresh_sim_tensors()
        return

    def create_obj(self, env_id, obj_type, asset_file, name, is_visual=False, enable_self_collisions=True, 
                     fix_root=False, start_pos=None, start_rot=None,
                     color=None, disable_motors=False):
        segmentation_id = 0

        _, file_ext = os.path.splitext(asset_file)
        assert(file_ext == ".xml"), "Unsupported asset format for Isaac Gym: {:s}".format(file_ext)

        start_pose = gymapi.Transform()

        if (start_pos is not None):
            start_pose.p = gymapi.Vec3(start_pos[0], start_pos[1], start_pos[2])
        else:
            start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)

        if (start_rot is not None):
            start_pose.r = gymapi.Quat(start_rot[0], start_rot[1], start_rot[2], start_rot[3])
        else:
            start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            
        if (is_visual):
            num_envs = self.get_num_envs()
            col_group = num_envs + env_id
        else:
            col_group = env_id

        if (enable_self_collisions):
            col_filter = 0
        else:
            col_filter = 1
            
        env_ptr = self.get_env(env_id)
        asset = self._load_asset(asset_file, fix_root)
        obj_id = self._gym.create_actor(env_ptr, asset, start_pose, name, col_group, col_filter, segmentation_id)
        
        if (self._enable_dof_force_sensors()):
            self._gym.enable_actor_dof_force_sensors(env_ptr, obj_id)

        if (disable_motors):
            control_mode = engine.ControlMode.none
        else:
            control_mode = self.get_control_mode()

        dof_props = self._gym.get_actor_dof_properties(env_ptr, obj_id)
        kp = dof_props["stiffness"]
        kd = dof_props["damping"]
        
        actuator_props = self._gym.get_actor_actuator_properties(env_ptr, obj_id)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        torque_lim = np.array(motor_efforts)
        
        if (control_mode == engine.ControlMode.none):
            kp = np.zeros_like(kp)
            kd = np.zeros_like(kd)

        self._obj_kp[env_id].append(torch.tensor(kp, device=self._device, dtype=torch.float32))
        self._obj_kd[env_id].append(torch.tensor(kd, device=self._device, dtype=torch.float32))
        self._obj_torque_lim[env_id].append(torch.tensor(torque_lim, device=self._device, dtype=torch.float32))
        assert(obj_id == len(self._obj_kp[env_id]) - 1)
        assert(obj_id == len(self._obj_kd[env_id]) - 1)
        assert(obj_id == len(self._obj_torque_lim[env_id]) - 1)

        drive_mode = self._control_mode_to_drive_mode(control_mode)
        dof_props["driveMode"] = drive_mode
        self._modify_control_mode_dof_props(control_mode, dof_props)

        self._gym.set_actor_dof_properties(env_ptr, obj_id, dof_props)

        if (color is not None):
            num_bodies = self.get_obj_num_bodies(obj_id)
            for b in range(num_bodies):
                self.set_body_color(env_id, obj_id, b, color)

        self._obj_types[env_id].append(obj_type)

        return obj_id
        
    def set_cmd(self, obj_id, cmd):
        if (self._has_dof()):
            obj_cmd = self._obj_dof_cmd[obj_id]
            obj_cmd[:] = cmd
        return
    
    def set_camera_pose(self, pos, look_at):
        self._gym.viewer_camera_look_at(self._viewer, None, 
                                      gymapi.Vec3(pos[0], pos[1], pos[2]),
                                      gymapi.Vec3(look_at[0], look_at[1], look_at[2]))
        return
    
    def get_camera_pos(self):
        cam_trans = self._gym.get_viewer_camera_transform(self._viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        return cam_pos
    
    def get_camera_dir(self):
        cam_trans = self._gym.get_viewer_camera_transform(self._viewer, None)
        cam_rot = torch.tensor([cam_trans.r.x, cam_trans.r.y, cam_trans.r.z, cam_trans.r.w], device="cpu")
        ref_dir = torch.tensor([0.0, 0.0, 1.0], device="cpu")
        cam_dir = torch_util.quat_rotate(cam_rot, ref_dir)
        cam_dir = cam_dir.numpy()
        return cam_dir
    
    def render(self):
        self._process_gui_events()

        # fetch results
        if (self._device != "cpu"):
            self._gym.fetch_results(self._sim, True)
        
        self._gym.step_graphics(self._sim)
        self._gym.draw_viewer(self._viewer, self._sim, True)
        self._gym.clear_lines(self._viewer)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        self._gym.sync_frame_time(self._sim)

        # it seems that in some cases sync_frame_time still results in higher-than-realtime framerate
        # this code will slow down the rendering to real time
        now = time.time()
        delta = now - self._prev_frame_time
        time_step = self.get_timestep()

        if (delta < time_step):
            time.sleep(time_step - delta)

        self._prev_frame_time = time.time()
        return
    
    def get_timestep(self):
        return self._timestep
    
    def get_sim_timestep(self):
        sim_params = self._gym.get_sim_params(self._sim)
        return sim_params.dt
    
    def get_num_envs(self):
        return self._num_envs
    
    def get_objs_per_env(self):
        return self._objs_per_env
    
    def get_root_pos(self, obj_id):
        return self._root_pos[..., obj_id, :]
    
    def get_root_rot(self, obj_id):
        return self._root_rot[..., obj_id, :]
    
    def get_root_vel(self, obj_id):
        return self._root_vel[..., obj_id, :]
    
    def get_root_ang_vel(self, obj_id):
        return self._root_ang_vel[..., obj_id, :]
    
    def get_dof_pos(self, obj_id):
        return self._obj_dof_pos[obj_id]
    
    def get_dof_vel(self, obj_id):
        return self._obj_dof_vel[obj_id]
    
    def get_dof_forces(self, obj_id):
        return self._obj_dof_forces[obj_id]
    
    def get_body_pos(self, obj_id):
        return self._obj_body_pos[obj_id]
    
    def get_body_rot(self, obj_id):
        return self._obj_body_rot[obj_id]
    
    def get_body_vel(self, obj_id):
        return self._obj_body_vel[obj_id]
    
    def get_body_ang_vel(self, obj_id):
        return self._obj_body_ang_vel[obj_id]
    
    def get_contact_forces(self, obj_id):
        return self._obj_contact_forces[obj_id]
    
    def get_ground_contact_forces(self, obj_id):
        ground_forces = self._obj_contact_forces[obj_id].clone()
        body_pos = self.get_body_pos(obj_id)

        body_height = body_pos[..., 2]
        above_ground = body_height > self._ground_contact_height
        ground_forces[above_ground, :] = 0.0

        return ground_forces
    
    def set_root_pos(self, env_id, obj_id, root_pos):
        if (env_id is None):
            self._root_pos[:, obj_id, :] = root_pos
        else:
            self._root_pos[env_id, obj_id, :] = root_pos

        self._flag_obj_needs_reset(env_id, obj_id)
        return
    
    def set_root_rot(self, env_id, obj_id, root_rot):
        if (env_id is None):
            self._root_rot[:, obj_id, :] = root_rot
        else:
            self._root_rot[env_id, obj_id, :] = root_rot
        
        self._flag_obj_needs_reset(env_id, obj_id)
        return
    
    def set_root_vel(self, env_id, obj_id, root_vel):
        if (env_id is None):
            self._root_vel[:, obj_id, :] = root_vel
        else:
            self._root_vel[env_id, obj_id, :] = root_vel
        
        self._flag_obj_needs_reset(env_id, obj_id)
        return
    
    def set_root_ang_vel(self, env_id, obj_id, root_ang_vel):
        if (env_id is None):
            self._root_ang_vel[:, obj_id, :] = root_ang_vel
        else:
            self._root_ang_vel[env_id, obj_id, :] = root_ang_vel
        
        self._flag_obj_needs_reset(env_id, obj_id)
        return
    
    def set_dof_pos(self, env_id, obj_id, dof_pos):
        if (env_id is None):
            self._obj_dof_pos[obj_id][:] = dof_pos
        else:
            self._obj_dof_pos[obj_id][env_id, :] = dof_pos
        
        self._flag_obj_needs_reset(env_id, obj_id)
        return
    
    def set_dof_vel(self, env_id, obj_id, dof_vel):
        if (env_id is None):
            self._obj_dof_vel[obj_id][:] = dof_vel
        else:
            self._obj_dof_vel[obj_id][env_id, :] = dof_vel
        
        self._flag_obj_needs_reset(env_id, obj_id)
        return
    
    def set_body_pos(self, env_id, obj_id, body_pos):
        if (env_id is None):
            self._obj_body_pos[obj_id][:] = body_pos
        else:
            self._obj_body_pos[obj_id][env_id, :] = body_pos

        # don't need to flag reset after setting body states since those 
        # do not directly affect the simulator
        return
    
    def set_body_rot(self, env_id, obj_id, body_rot):
        if (env_id is None):
            self._obj_body_rot[obj_id][:] = body_rot
        else:
            self._obj_body_rot[obj_id][env_id, :] = body_rot
        return
    
    def set_body_vel(self, env_id, obj_id, body_vel):
        if (env_id is None):
            self._obj_body_vel[obj_id][:] = body_vel
        else:
            self._obj_body_vel[obj_id][env_id, :] = body_vel
        return
    
    def set_body_ang_vel(self, env_id, obj_id, body_ang_vel):
        if (env_id is None):
            self._obj_body_ang_vel[obj_id][:] = body_ang_vel
        else:
            self._obj_body_ang_vel[obj_id][env_id, :] = body_ang_vel
        return
    
    def set_body_forces(self, env_id, obj_id, body_id, forces):
        if (env_id is None):
            self._obj_body_forces[obj_id][:, body_id, :] = forces
        else:
            self._obj_body_forces[obj_id][env_id, body_id, :] = forces

        if (env_id is None or len(env_id) > 0):
            self._has_body_forces = True
        return
    
    def get_obj_torque_limits(self, env_id, obj_id):
        return self._obj_torque_lim[obj_id][env_id].cpu().numpy()
    
    def get_obj_dof_limits(self, env_id, obj_id):
        env_ptr = self.get_env(env_id)
        dof_props = self._gym.get_actor_dof_properties(env_ptr, obj_id)
        dof_low = dof_props["lower"]
        dof_high = dof_props["upper"]

        low_arr = np.asarray(dof_low)
        high_arr = np.asarray(dof_high)

        # Sanity checks per-DOF - raise exceptions for bad limits
        threshold = 1e8
        for i, (l, h) in enumerate(zip(low_arr, high_arr)):
            # both bounds zero
            if l == 0 and h == 0:
                raise ValueError(f"Env {env_id} Obj {obj_id} DoF {i}: both lower and upper limits are 0.0 — this may indicate a fixed joint or missing limits. Either upper or lower limit must be non-zero for all DoFs.")

            # infinite or NaN bounds
            if not np.isfinite(l) or not np.isfinite(h):
                raise ValueError(f"Env {env_id} Obj {obj_id} DoF {i}: invalid bound detected (lower={l}, upper={h}).")

            # extremely large magnitude bounds (likely a placeholder for +/-inf)
            if (abs(l) > threshold or abs(h) > threshold):
                raise ValueError(f"Env {env_id} Obj {obj_id} DoF {i}: invalid bound detected (lower={l}, upper={h}).")

        return dof_low, dof_high
    
    def find_obj_body_id(self, obj_id, body_name):
        env_ptr = self.get_env(0)
        return self._gym.find_actor_rigid_body_index(env_ptr, obj_id, body_name, gymapi.DOMAIN_ACTOR)
    
    def get_env(self, env_id):
        return self._envs[env_id]
    
    def set_body_color(self, env_id, obj_id, body_id, color):
        env_ptr = self.get_env(env_id)
        self._gym.set_rigid_body_color(env_ptr, obj_id, body_id, gymapi.MESH_VISUAL,
                                       gymapi.Vec3(color[0], color[1], color[2]))
        return
    
    def get_obj_type(self, obj_id):
        return self._obj_types[0][obj_id]
    
    def get_obj_num_bodies(self, obj_id):
        env_ptr = self.get_env(0)
        num_bodies = self._gym.get_actor_rigid_body_count(env_ptr, obj_id)
        return num_bodies
    
    def get_obj_num_dofs(self, obj_id):
        env_ptr = self.get_env(0)
        num_dofs = self._gym.get_actor_dof_count(env_ptr, obj_id)
        return num_dofs
    
    def get_obj_body_names(self, obj_id):
        env_ptr = self.get_env(0)
        body_names = self._gym.get_actor_rigid_body_names(env_ptr, obj_id)
        return body_names
    
    def calc_obj_mass(self, env_id, obj_id):
        env_ptr = self.get_env(env_id)
        rb_props = self._gym.get_actor_rigid_body_properties(env_ptr, obj_id)
        total_mass = sum(rb.mass for rb in rb_props)
        return total_mass
    
    def get_control_mode(self):
        return self._control_mode
    
    def draw_lines(self, env_id, start_verts, end_verts, cols, line_width):
        env_ptr = self.get_env(env_id)
        num_lines = start_verts.shape[0]
        verts = np.concatenate([start_verts, end_verts], axis=-1)
        cols = cols[..., :3]
        
        self._gym.add_lines(self._viewer, env_ptr, num_lines, verts, cols)
        return
    
    def register_keyboard_callback(self, key_str, callback_func):
        assert(key_str not in self._keyboard_callbacks)

        key_code = str_to_key_code(key_str)
        self._gym.subscribe_viewer_keyboard_event(self._viewer, key_code, key_str)
        self._keyboard_callbacks[key_str] = callback_func
        return

    def _load_asset(self, file, fix_root):
        asset_cfg = AssetCfg(asset_file=file, fix_root=fix_root)

        if (asset_cfg in self._asset_cache):
            asset = self._asset_cache[asset_cfg]
        else:
            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.max_linear_velocity = 1000.0
            asset_options.max_angular_velocity = 1000.0
            asset_options.fix_base_link = fix_root

            file_dir = os.path.dirname(file)
            file_name = os.path.basename(file)
            asset = self._gym.load_asset(self._sim, file_dir, file_name, asset_options)

            self._asset_cache[asset_cfg] = asset

        return asset
    
    def _build_ground(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.0

        self._gym.add_ground(self._sim, plane_params)
        return
    
    def _create_simulator(self, sim_timestep, visualize):
        physics_engine = gymapi.SimType.SIM_PHYSX
        sim_params = self._build_sim_params(sim_timestep)

        compute_device_id = self._get_device_idx()
        if (visualize):
            graphics_device_id = compute_device_id
        else:
            graphics_device_id = -1

        sim = self._gym.create_sim(compute_device_id, graphics_device_id, 
                                   physics_engine, sim_params)
        assert(sim is not None), "Failed to create simulator."
        return sim

    def _apply_cmd(self):
        if (self._has_dof()):
            if (self._control_mode == engine.ControlMode.none):
                pass
            elif (self._control_mode == engine.ControlMode.pos):
                cmd_buf = self._get_dof_cmd_buf()
                cmd_tensor = gymtorch.unwrap_tensor(cmd_buf)
                self._gym.set_dof_position_target_tensor(self._sim, cmd_tensor)

            elif (self._control_mode == engine.ControlMode.vel):
                cmd_buf = self._get_dof_cmd_buf()
                cmd_tensor = gymtorch.unwrap_tensor(cmd_buf)
                self._gym.set_dof_velocity_target_tensor(self._sim, cmd_tensor)
                
            elif (self._control_mode == engine.ControlMode.torque):
                pass
            elif (self._control_mode == engine.ControlMode.pd_explicit):
                pass
            else:
                assert(False), "Unsupported control mode: {}".format(self._control_mode)
        return

    def _pre_sim_step(self):
        if (self._control_mode == engine.ControlMode.pd_explicit):
            self._gym.refresh_dof_state_tensor(self._sim)

        if (self._control_mode == engine.ControlMode.torque):
            cmd_buf = self._get_dof_cmd_buf()
            self._set_actuation_torque(cmd_buf)

        elif (self._control_mode == engine.ControlMode.pd_explicit):
            torque = self._calc_pd_explicit_torque()
            self._set_actuation_torque(torque)

        if (self._has_body_forces):
            self._gym.apply_rigid_body_force_tensors(self._sim, gymtorch.unwrap_tensor(self._body_forces_raw), 
                                                     None, gymapi.CoordinateSpace.GLOBAL_SPACE)
        return

    def _sim_step(self):
        self._gym.simulate(self._sim)
        return

    def _set_actuation_torque(self, torque):
        torque_clip = torch.clip(torque, -self._torque_lim_raw, self._torque_lim_raw)
        torque_tensor = gymtorch.unwrap_tensor(torque_clip)
        self._gym.set_dof_actuation_force_tensor(self._sim, torque_tensor)
        return
    
    def _flag_obj_needs_reset(self, env_id, obj_id):
        if (env_id is None):
            self._objs_need_reset[:, obj_id] = True
        elif (len(env_id) > 0):
            self._objs_need_reset[env_id, obj_id] = True
        return
    
    def _update_reset_objs(self):
        obj_ids = self._need_reset_buf.nonzero(as_tuple=False)
        obj_ids = obj_ids.type(torch.int32).flatten()

        if (len(obj_ids) > 0):
            self._gym.set_actor_root_state_tensor_indexed(self._sim,
                                                         gymtorch.unwrap_tensor(self._root_state_raw),
                                                         gymtorch.unwrap_tensor(obj_ids), len(obj_ids))
            if (self._has_dof()):
                has_dof = self._obj_dof_dims[obj_ids.type(torch.long)] > 0
                dof_obj_ids = obj_ids[has_dof]
                
                if (dof_obj_ids.shape[0] > 0):
                    self._gym.set_dof_state_tensor_indexed(self._sim,
                                                        gymtorch.unwrap_tensor(self._dof_state_raw),
                                                        gymtorch.unwrap_tensor(dof_obj_ids), len(dof_obj_ids))
                    
                    dof_pos = self._dof_state_raw[..., :, 0]
                    dof_pos = dof_pos.contiguous()
                    self._gym.set_dof_position_target_tensor_indexed(self._sim,
                                                        gymtorch.unwrap_tensor(dof_pos),
                                                        gymtorch.unwrap_tensor(dof_obj_ids), len(dof_obj_ids))

            self._objs_need_reset[:] = False
        return
    
    def _refresh_sim_tensors(self):
        if (self._device == "cpu"):
            self._gym.fetch_results(self._sim, True)
        
        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_rigid_body_state_tensor(self._sim)

        self._gym.refresh_force_sensor_tensor(self._sim)
        self._gym.refresh_net_contact_force_tensor(self._sim)
        
        if self._enable_dof_force_sensors():
            self._gym.refresh_dof_force_tensor(self._sim)
            
        if (self._has_body_forces):
            self._body_forces_raw[:] = 0.0
            self._has_body_forces = False
        return
    
    def _control_mode_to_drive_mode(self, mode):
        if (mode == engine.ControlMode.none):
            drive_mode = gymapi.DOF_MODE_NONE
        elif (mode == engine.ControlMode.pos):
            drive_mode = gymapi.DOF_MODE_POS
        elif (mode == engine.ControlMode.vel):
            drive_mode = gymapi.DOF_MODE_VEL
        elif (mode == engine.ControlMode.torque):
            drive_mode = gymapi.DOF_MODE_EFFORT   
        elif (mode == engine.ControlMode.pd_explicit):
            drive_mode = gymapi.DOF_MODE_EFFORT 
        else:
            assert(False), "Unsupported control mode: {}".format(mode)
        return drive_mode
    
    def _build_sim_params(self, sim_timestep):
        sim_params = gymapi.SimParams()
        sim_params.substeps = 1
        sim_params.dt = sim_timestep
        sim_params.num_client_threads = 0
        
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity.x = 0
        sim_params.gravity.y = 0
        sim_params.gravity.z = -9.81
        
        sim_params.physx.num_threads = 4
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.contact_offset = 0.02
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.bounce_threshold_velocity = 0.2
        sim_params.physx.max_depenetration_velocity = 10.0
        sim_params.physx.default_buffer_size_multiplier = 10.0
        sim_params.physx.num_subscenes = 0
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
        
        sim_params.flex.dynamic_friction = 1.0
        sim_params.flex.static_friction = 1.0
        sim_params.flex.num_inner_iterations = 10
        sim_params.flex.warm_start = 0.25

        if ("gpu" in self._device or "cuda" in self._device):
            sim_params.physx.use_gpu = True
            sim_params.use_gpu_pipeline = True
        elif ("cpu" in self._device):
            sim_params.physx.use_gpu = False
            sim_params.use_gpu_pipeline = False
        else:
            assert(False), "Unsupported simulation device: {}".format(self._device)

        return sim_params
    
    def _get_device_idx(self):
        re_idx = re.search(r"\d", self._device)
        if (re_idx is None):
            device_idx = 0
        else:
            num_idx = re_idx.start()
            device_idx = int(self._device[num_idx:])
        return device_idx
    
    def _modify_control_mode_dof_props(self, control_mode, dof_props):
        if (control_mode == engine.ControlMode.none):
            dof_props["stiffness"] = 0.0
            dof_props["damping"] = 0.0
        elif (control_mode == engine.ControlMode.pos):
            pass
        elif (control_mode == engine.ControlMode.vel):
            dof_props["stiffness"] = 0.0
        elif (control_mode == engine.ControlMode.torque):
            dof_props["stiffness"] = 0.0
            dof_props["damping"] = 0.0
        elif (control_mode == engine.ControlMode.pd_explicit):
            dof_props["stiffness"] = 0.0
            dof_props["damping"] = 0.0
        else:
            assert(False), "Unsupported control mode: {}".format(control_mode)
        return

    def _build_sim_tensors(self):
        root_state_tensor = self._gym.acquire_actor_root_state_tensor(self._sim)
        dof_state_tensor = self._gym.acquire_dof_state_tensor(self._sim)
        body_state_tensor = self._gym.acquire_rigid_body_state_tensor(self._sim)
        contact_force_tensor = self._gym.acquire_net_contact_force_tensor(self._sim)
        
        self._root_state_raw = gymtorch.wrap_tensor(root_state_tensor)
        self._dof_state_raw = gymtorch.wrap_tensor(dof_state_tensor)
        self._body_state_raw = gymtorch.wrap_tensor(body_state_tensor)
        self._contact_force_raw = gymtorch.wrap_tensor(contact_force_tensor)
        
        num_envs = self.get_num_envs()
        num_objs = self._root_state_raw.shape[0]
        self._objs_per_env = num_objs // num_envs

        root_state = self._root_state_raw.view([num_envs, self._objs_per_env, 
                                                self._root_state_raw.shape[-1]])
        
        self._root_state = root_state
        self._root_pos = root_state[..., 0:3]
        self._root_rot = root_state[..., 3:7]
        self._root_vel = root_state[..., 7:10]
        self._root_ang_vel = root_state[..., 10:13]

        if (self._dof_state_raw is not None):
            dofs_per_env = self._dof_state_raw.shape[0] // num_envs
            dof_state = self._dof_state_raw.view([num_envs, dofs_per_env, 2])
            self._dof_state = dof_state
            self._obj_dof_pos = []
            self._obj_dof_vel = []
            
            if self._enable_dof_force_sensors():
                dof_force_tensor = self._gym.acquire_dof_force_tensor(self._sim)
                self._dof_forces_raw = gymtorch.wrap_tensor(dof_force_tensor)
                dof_forces = self._dof_forces_raw.view([num_envs, dofs_per_env])
                self._obj_dof_forces = []

            dof_idx0 = 0
            for i in range(self._objs_per_env):
                obj_dof_dim = self.get_obj_num_dofs(i)
                dof_idx1 = dof_idx0 + obj_dof_dim
                obj_dof_pos = dof_state[..., dof_idx0:dof_idx1, 0]
                obj_dof_vel = dof_state[..., dof_idx0:dof_idx1, 1]

                self._obj_dof_pos.append(obj_dof_pos)
                self._obj_dof_vel.append(obj_dof_vel)
                
                if self._enable_dof_force_sensors():
                    obj_dof_force = dof_forces[..., dof_idx0:dof_idx1]
                    self._obj_dof_forces.append(obj_dof_force)

                dof_idx0 = dof_idx1

            self._build_control_tensors()
        
        else:
            self._dof_state_raw = None

        bodies_per_env = self._body_state_raw.shape[0] // num_envs
        body_state = self._body_state_raw.view([num_envs, bodies_per_env, 
                                                self._body_state_raw.shape[-1]])

        contact_forces = self._contact_force_raw.view([num_envs, bodies_per_env, 3])

        self._body_forces_raw = torch.zeros_like(self._body_state_raw[..., :3])
        body_forces = self._body_forces_raw.view([num_envs, bodies_per_env, 3])

        # Note: Careful when using these tensors in observations calculations
        # they are not updated immediately during episode resets, and are not valid
        # until the after the simulator as been updated for one timestep
        self._obj_body_pos = []
        self._obj_body_rot = []
        self._obj_body_vel = []
        self._obj_body_ang_vel = []
        self._obj_contact_forces = []
        self._obj_body_forces = []
        self._has_body_forces = False

        body_idx0 = 0
        for obj_id in range(self._objs_per_env):
            num_bodies = self.get_obj_num_bodies(obj_id)
            body_idx1 = body_idx0 + num_bodies

            obj_body_pos = body_state[..., body_idx0:body_idx1, 0:3]
            obj_body_rot = body_state[..., body_idx0:body_idx1, 3:7]
            obj_body_vel = body_state[..., body_idx0:body_idx1, 7:10]
            obj_body_ang_vel = body_state[..., body_idx0:body_idx1, 10:13]
            obj_body_forces = body_forces[..., body_idx0:body_idx1, :]
            obj_contact_forces = contact_forces[..., body_idx0:body_idx1, :]

            self._obj_body_pos.append(obj_body_pos)
            self._obj_body_rot.append(obj_body_rot)
            self._obj_body_vel.append(obj_body_vel)
            self._obj_body_ang_vel.append(obj_body_ang_vel)
            self._obj_body_forces.append(obj_body_forces)
            self._obj_contact_forces.append(obj_contact_forces)

            body_idx0 = body_idx1
        
        self._need_reset_buf = torch.zeros(num_objs, device=self._device, dtype=torch.bool)
        self._objs_need_reset = self._need_reset_buf.view((num_envs, self._objs_per_env))
        self._obj_dof_dims = self._build_obj_dof_dims()
        return
    
    def _build_control_tensors(self):
        self._kp_raw = [torch.cat(kp) for kp in self._obj_kp]
        self._kp_raw = torch.stack(self._kp_raw, dim=0)
        
        self._kd_raw = [torch.cat(kd) for kd in self._obj_kd]
        self._kd_raw = torch.stack(self._kd_raw, dim=0)
        
        self._torque_lim_raw = [torch.cat(lim) for lim in self._obj_torque_lim]
        self._torque_lim_raw = torch.stack(self._torque_lim_raw, dim=0)

        self._dof_cmd_raw = torch.zeros_like(self._kp_raw)

        self._obj_kp = []
        self._obj_kd = []
        self._obj_torque_lim = []
        self._obj_dof_cmd = []

        dof_idx0 = 0
        objs_per_env = self.get_objs_per_env()

        for i in range(objs_per_env):
            obj_dof_dim = self.get_obj_num_dofs(i)
            dof_idx1 = dof_idx0 + obj_dof_dim
            obj_kp = self._kp_raw[..., dof_idx0:dof_idx1]
            obj_kd = self._kd_raw[..., dof_idx0:dof_idx1]
            obj_torque_lim = self._torque_lim_raw[..., dof_idx0:dof_idx1]
            obj_dof_cmd = self._dof_cmd_raw[..., dof_idx0:dof_idx1]

            self._obj_kp.append(obj_kp)
            self._obj_kd.append(obj_kd)
            self._obj_torque_lim.append(obj_torque_lim)
            self._obj_dof_cmd.append(obj_dof_cmd)

            dof_idx0 = dof_idx1

        return
    
    def _build_obj_dof_dims(self):
        num_envs = self.get_num_envs()
        objs_per_env = self.get_objs_per_env()

        obj_dof_dims = torch.zeros(num_envs * objs_per_env, device=self._device, dtype=torch.int)
        env_obj_dof_dims = obj_dof_dims.view([num_envs, objs_per_env])

        for obj_id in range(objs_per_env):
            num_dofs = self.get_obj_num_dofs(obj_id)
            env_obj_dof_dims[:, obj_id] = num_dofs

        return obj_dof_dims
    
    def _enable_dof_force_sensors(self):
        enable_dof_force_sensors = False
        return (enable_dof_force_sensors and self._has_dof())
    
    def _has_dof(self):
        return self._dof_state_raw is not None

    def _get_env_spacing(self):
        return self._env_spacing

    def _get_dof_cmd_buf(self):
        return self._dof_cmd_raw
    
    def _calc_pd_explicit_torque(self):
        dof_pos = self._dof_state[..., :, 0]
        dof_vel = self._dof_state[..., :, 1]
        tar_dof = self._get_dof_cmd_buf()

        torque = self._kp_raw * (tar_dof - dof_pos) - self._kd_raw * dof_vel
        return torque
    
    def _build_viewer(self):
        # subscribe to keyboard shortcuts
        self._viewer = self._gym.create_viewer(self._sim, gymapi.CameraProperties())
        self._gym.subscribe_viewer_keyboard_event(self._viewer, gymapi.KEY_ESCAPE, "QUIT")

        # set the camera position based on up axis
        sim_params = self._gym.get_sim_params(self._sim)
        cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
        cam_target = gymapi.Vec3(10.0, 15.0, 0.0)

        self._gym.viewer_camera_look_at(self._viewer, None, cam_pos, cam_target)
        return
    
    def _process_gui_events(self):
        # check for window closed
        if (self._gym.query_viewer_has_closed(self._viewer)):
            sys.exit()

        # check for keyboard events
        for evt in self._gym.query_viewer_action_events(self._viewer):
            if (evt.value > 0):
                if (evt.action == "QUIT"):
                    sys.exit()
                elif (evt.action in self._keyboard_callbacks):
                    callback = self._keyboard_callbacks[evt.action]
                    callback()
        return