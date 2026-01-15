import warp as wp
wp.config.enable_backward = False

import newton
import numpy as np
import os
import torch
import pyglet

import engines.engine as engine
from util.logger import Logger

def str_to_key_code(key_str):
    key_name = key_str.upper()

    if (len(key_str) == 1 and key_str.isdigit()):
        key_name = "_" + key_str
    elif (key_str == "ESC"):
        key_name = "ESCAPE"
    elif (key_str == "RETURN"):
        key_name = "ENTER"
    elif (key_str == "DELETE"):
        key_name = "DEL"
    elif (key_str == "LEFT_SHIFT"):
        key_name = "LSHIFT"
    elif (key_str == "LEFT_CONTROL"):
        key_name = "LCTRL"
    elif (key_str == "LEFT_ALT"):
        key_name = "LALT"
    elif (key_str == "RIGHT_SHIFT"):
        key_name = "RSHIFT"
    elif (key_str == "RIGHT_CONTROL"):
        key_name = "RCTRL"
    elif (key_str == "RIGHT_ALT"):
        key_name = "RALT"
    
    key_code = getattr(pyglet.window.key, key_name)
    return key_code

class SimState:
    def __init__(self, sim_model, num_envs):
        self._sim_model = sim_model
        self.raw_state = self._sim_model.state()
        self.eval_fk()

        self._torch_joint_q = wp.to_torch(self.raw_state.joint_q)
        self._torch_joint_qd = wp.to_torch(self.raw_state.joint_qd)
        self._torch_body_q = wp.to_torch(self.raw_state.body_q)
        self._torch_body_qd = wp.to_torch(self.raw_state.body_qd)
        
        self._wp_body_force = wp.clone(self.raw_state.body_f)
        self._torch_body_force = wp.to_torch(self._wp_body_force)

        # the joint parent transform is only used for objects with a fixed root
        self._torch_joint_X_p = wp.to_torch(sim_model.joint_X_p)
        
        self.root_pos = []
        self.root_rot = []
        self.root_vel = []
        self.root_ang_vel = []
        self.dof_pos = []
        self.dof_vel = []

        self.body_pos = []
        self.body_rot = []
        self.body_vel = []
        self.body_ang_vel = []
        self.body_force = []

        articulation_start = sim_model.articulation_start.numpy()
        joint_q_start = sim_model.joint_q_start.numpy()
        joint_qd_start = sim_model.joint_qd_start.numpy()
        joint_types = self._sim_model.joint_type.numpy()

        num_objs = sim_model.articulation_count
        objs_per_env = int(num_objs / num_envs)

        joint_q = self._torch_joint_q.view(num_envs, -1)
        joint_qd = self._torch_joint_qd.view(num_envs, -1)
        body_q = self._torch_body_q.view(num_envs, -1, 7)
        body_qd = self._torch_body_qd.view(num_envs, -1, 6)
        body_force = self._torch_body_force.view(num_envs, -1, 6)
        joint_X_p = self._torch_joint_X_p.view(num_envs, -1, 7)

        self._record_spherical_joints()
        has_sphere_joints = self._has_spherical_joints()

        if (has_sphere_joints):
            self._wp_dof_pos = wp.clone(self.raw_state.joint_qd)
            self._torch_dof_pos = wp.to_torch(self._wp_dof_pos)
            dof_pos = self._torch_dof_pos.view(num_envs, -1)

        for obj_id in range(objs_per_env):
            body_start = articulation_start[obj_id]
            body_end = articulation_start[obj_id + 1]

            q_root_start = joint_q_start[body_start]
            q_dof_start = joint_q_start[body_start + 1]
            q_dof_end = joint_q_start[body_end]

            qd_root_start = joint_qd_start[body_start]
            qd_dof_start = joint_qd_start[body_start + 1]
            qd_dof_end = joint_qd_start[body_end]

            obj_body_q = body_q[:, body_start:body_end, :]
            obj_body_qd = body_qd[:, body_start:body_end, :]
            obj_body_force = body_force[:, body_start:body_end, :]

            root_type = joint_types[body_start]
            fix_root = root_type == newton.JointType.FIXED

            if (fix_root):
                obj_root_pos = joint_X_p[..., body_start, :3]
                obj_root_rot = joint_X_p[..., body_start, 3:7]
                obj_root_vel = obj_body_qd[:, 0, :3]
                obj_root_ang_vel = obj_body_qd[:, 0, 3:6]
            else:
                obj_root_pos = joint_q[:, q_root_start:q_root_start + 3]
                obj_root_rot = joint_q[:, q_root_start + 3:q_root_start + 7]
                obj_root_vel = joint_qd[:, qd_root_start:qd_root_start + 3]
                obj_root_ang_vel = joint_qd[:, qd_root_start + 3:qd_root_start + 6]

            if (has_sphere_joints):
                obj_dof_pos = dof_pos[:, qd_dof_start:qd_dof_end]
            else:
                obj_dof_pos = joint_q[:, q_dof_start:q_dof_end]
            
            obj_dof_vel = joint_qd[:, qd_dof_start:qd_dof_end]

            obj_body_pos = obj_body_q[..., :3]
            obj_body_rot = obj_body_q[..., 3:]
            obj_body_vel = obj_body_qd[..., :3]
            obj_body_ang_vel = obj_body_qd[..., 3:]

            self.root_pos.append(obj_root_pos)
            self.root_rot.append(obj_root_rot)
            self.dof_pos.append(obj_dof_pos)

            self.dof_vel.append(obj_dof_vel)
            self.root_vel.append(obj_root_vel)
            self.root_ang_vel.append(obj_root_ang_vel)

            self.body_pos.append(obj_body_pos)
            self.body_rot.append(obj_body_rot)
            self.body_vel.append(obj_body_vel)
            self.body_ang_vel.append(obj_body_ang_vel)
            self.body_force.append(obj_body_force)

        return
    
    def clear_forces(self):
        self._wp_body_force.zero_()
        return
    
    def copy(self, raw_state):
        wp.copy(self.raw_state.joint_q, raw_state.joint_q)
        wp.copy(self.raw_state.joint_qd, raw_state.joint_qd)
        wp.copy(self.raw_state.body_q, raw_state.body_q)
        wp.copy(self.raw_state.body_qd, raw_state.body_qd)
        wp.copy(self.raw_state.body_f, raw_state.body_f)
        return
    
    def pre_step_update(self):
        if (self._has_spherical_joints()):
            wp.launch(
                kernel=copy_indexed,
                dim=self._wp_nonsphere_joint_dof_idx.shape[0],
                inputs=[self._wp_dof_pos, self._wp_nonsphere_joint_dof_idx, 
                        self._wp_nonsphere_joint_q_idx, self.raw_state.joint_q]
            )
            wp.launch(
                kernel=exp_map_to_quat_indexed,
                dim=self._wp_sphere_dof_start.shape[0],
                inputs=[self._wp_dof_pos, self._wp_sphere_dof_start, 
                        self._wp_sphere_q_start, self.raw_state.joint_q]
            )
        return

    def post_step_update(self):
        if (self._has_spherical_joints()):
            wp.launch(
                kernel=copy_indexed,
                dim=self._wp_nonsphere_joint_q_idx.shape[0],
                inputs=[self.raw_state.joint_q, self._wp_nonsphere_joint_q_idx, 
                        self._wp_nonsphere_joint_dof_idx, self._wp_dof_pos]
            )
            wp.launch(
                kernel=quat_to_exp_map_indexed,
                dim=self._wp_sphere_q_start.shape[0],
                inputs=[self.raw_state.joint_q, self._wp_sphere_q_start, 
                        self._wp_sphere_dof_start, self._wp_dof_pos]
            )
        return
    
    def eval_fk(self):
        newton.eval_fk(self._sim_model, self.raw_state.joint_q, self.raw_state.joint_qd, self.raw_state)
        return
    
    def _record_spherical_joints(self):
        sphere_type = newton.JointType.BALL

        joint_q_start = self._sim_model.joint_q_start.numpy()
        joint_qd_start = self._sim_model.joint_qd_start.numpy()
        joint_types = self._sim_model.joint_type.numpy()
        joint_parent = self._sim_model.joint_parent.numpy()
        num_joints = self._sim_model.joint_count

        sphere_q_start = []
        sphere_dof_start = []
        nonsphere_q_idx = []
        nonsphere_dof_idx = []

        for j in range(num_joints):
            curr_joint_type = joint_types[j]
            curr_joint_parent = joint_parent[j]
            is_root = curr_joint_parent == -1

            q_start = joint_q_start[j]
            q_end = joint_q_start[j + 1]
            dof_start = joint_qd_start[j]
            dof_end = joint_qd_start[j + 1]

            if (is_root):
                pass
            elif (curr_joint_type == sphere_type):
                assert(q_end - q_start == 4)
                assert(dof_end - dof_start == 3)
                sphere_q_start.append(q_start)
                sphere_dof_start.append(dof_start)
            else:
                curr_q_idx = [i for i in range(q_start, q_end)]
                curr_dof_idx = [i for i in range(dof_start, dof_end)]
                nonsphere_q_idx += curr_q_idx
                nonsphere_dof_idx += curr_dof_idx

        self._wp_sphere_q_start = wp.array(sphere_q_start, device=self._sim_model.device, dtype=int)
        self._wp_sphere_dof_start = wp.array(sphere_dof_start, device=self._sim_model.device, dtype=int)
        self._wp_nonsphere_joint_q_idx = wp.array(nonsphere_q_idx, device=self._sim_model.device, dtype=int)
        self._wp_nonsphere_joint_dof_idx = wp.array(nonsphere_dof_idx, device=self._sim_model.device, dtype=int)
        return
    
    def _has_spherical_joints(self):
        return self._wp_sphere_q_start.shape[0] > 0


class Controls:
    def __init__(self, sim_model, num_envs):
        self.control = sim_model.control()

        self._torch_target_pos = wp.to_torch(self.control.joint_target_pos)
        self._torch_target_vel = wp.to_torch(self.control.joint_target_vel)
        self._torch_joint_force = wp.to_torch(self.control.joint_f)

        target_pos = self._torch_target_pos.view(num_envs, -1)
        target_vel = self._torch_target_vel.view(num_envs, -1)
        joint_force = self._torch_joint_force.view(num_envs, -1)

        articulation_start = sim_model.articulation_start.numpy()
        joint_qd_start = sim_model.joint_qd_start.numpy()

        num_objs = sim_model.articulation_count
        objs_per_env = int(num_objs / num_envs)

        self.target_pos = []
        self.target_vel = []
        self.joint_force = []
        
        for obj_id in range(objs_per_env):
            body_start = articulation_start[obj_id]
            body_end = articulation_start[obj_id + 1]
            qd_start = joint_qd_start[body_start + 1]
            qd_end = joint_qd_start[body_end]

            obj_target_pos = target_pos[:, qd_start:qd_end]
            obj_target_vel = target_vel[:, qd_start:qd_end]
            obj_joint_force = joint_force[:, qd_start:qd_end]

            self.target_pos.append(obj_target_pos)
            self.target_vel.append(obj_target_vel)
            self.joint_force.append(obj_joint_force)

        return


class ObjCfg():
    def __init__(self, asset_file, fix_root, is_visual, enable_self_collisions):
        self.asset_file = asset_file
        self.fix_root = fix_root
        self.is_visual = is_visual
        self.enable_self_collisions = enable_self_collisions
        return
    
    def __hash__(self):
        values = (self.asset_file, self.fix_root, self.is_visual, self.enable_self_collisions)
        obj_hash = hash(values)
        return obj_hash
    
    def __eq__(self, other):
        same = self.asset_file == other.asset_file
        same &= self.fix_root == other.fix_root
        same &= self.is_visual == other.is_visual
        same &= self.enable_self_collisions == other.enable_self_collisions
        return same


class NewtonEngine(engine.Engine):
    def __init__(self, config, num_envs, device, visualize):
        super().__init__()

        self._device = device
        self._num_envs = num_envs
        self._env_spacing = config["env_spacing"]

        sim_freq = config.get("sim_freq", 60)
        control_freq = config.get("control_freq", 10)
        assert(sim_freq >= control_freq and sim_freq % control_freq == 0), \
            "Simulation frequency must be a multiple of the control frequency"

        self._timestep = 1.0 / control_freq
        self._sim_steps = int(sim_freq / control_freq)
        self._sim_timestep = 1.0 / sim_freq
        self._sim_step_count = 0

        self._scene_builder = self._create_model_builder()
        self._builder_cache = dict()

        self._obj_types = []
        self._obj_builders = []
        self._obj_start_pos = []
        self._obj_start_rot = []
        
        if ("control_mode" in config):
            self._control_mode = engine.ControlMode[config["control_mode"]]
        else:
            self._control_mode = engine.ControlMode.none

        self._build_ground()

        if (visualize):
            self._build_viewer()
            self._obj_colors = []
            self._keyboard_callbacks = dict()
        else:
            self._viewer = None

        return
    
    def get_name(self):
        return "newton"
    
    def create_env(self):
        env_id = len(self._obj_builders)
        assert(env_id < self.get_num_envs())

        self._obj_builders.append([])
        self._obj_types.append([])
        self._obj_start_pos.append([])
        self._obj_start_rot.append([])

        if (self._visualize()):
            self._obj_colors.append([])

        return env_id
    
    def initialize_sim(self):
        self._validate_envs()
        self._build_objs()

        Logger.print("Initializing simulation...")
        self._sim_model = self._scene_builder.finalize(device=self._device, requires_grad=False)
        
        num_envs = self.get_num_envs()
        self._sim_state = SimState(self._sim_model, num_envs)
        self._state_swap_buffer = self._sim_model.state()
        self._contacts = self._sim_model.collide(self._sim_state.raw_state)

        self._apply_start_xform()

        self._build_controls()
        self._build_contact_sensors()
        self._build_solver()

        self._build_sim_tensors()
        self._build_dof_force_tensors()

        if (self._visualize()):
            self._init_rendering()
        
        self._build_graphs()
        return
    
    def step(self):
        if (self._graph):
            wp.capture_launch(self._graph)
        else:
            self._simulate()

        self._update_contact_sensors()
        self._sim_step_count += 1
        return
    
    def create_obj(self, env_id, obj_type, asset_file, name, is_visual=False, enable_self_collisions=True, 
                   fix_root=False, start_pos=None, start_rot=None, color=None, disable_motors=False):
        if (start_rot is None):
            start_rot = np.array([0.0, 0.0, 0.0, 1.0])

        if (start_pos is None):
            start_pos = np.array([0.0, 0.0, 0.0])
        
        obj_builder = self._create_obj_builder(asset_file=asset_file, fix_root=fix_root, is_visual=is_visual, 
                                               enable_self_collisions=enable_self_collisions)
        
        obj_id = len(self._obj_builders[env_id])
        self._obj_types[env_id].append(obj_type)
        self._obj_builders[env_id].append(obj_builder)
        self._obj_start_pos[env_id].append(start_pos)
        self._obj_start_rot[env_id].append(start_rot)

        if (self._visualize()):
            self._obj_colors[env_id].append(color)
        
        return obj_id
    
    def set_cmd(self, obj_id, cmd):
        if (self._control_mode == engine.ControlMode.none):
            pass
        elif (self._control_mode == engine.ControlMode.pos):
            self._controls.target_pos[obj_id][:] = cmd
        elif (self._control_mode == engine.ControlMode.vel):
            self._controls.target_vel[obj_id][:] = cmd
        elif (self._control_mode == engine.ControlMode.torque):
            self._controls.joint_force[obj_id][:] = cmd
        elif (self._control_mode == engine.ControlMode.pd_explicit):
            self._controls.target_pos[obj_id][:] = cmd
        else:
            assert(False), "Unsupported control mode: {}".format(self._control_mode)
        return
    
    def set_camera_pose(self, pos, look_at):
        dx = look_at[0] - pos[0]
        dy = look_at[1] - pos[1]
        dz = look_at[2] - pos[2]

        pitch = np.arctan2(dz, np.sqrt(dx * dx + dy * dy))
        yaw = np.arctan2(dy, dx)
        cam_pos = pyglet.math.Vec3(pos[0] + self._camera_offsets[0, 0], 
                                   pos[1] + self._camera_offsets[0, 1], 
                                   pos[2] + self._camera_offsets[0, 2])
        self._viewer.set_camera(cam_pos, np.rad2deg(pitch), np.rad2deg(yaw))
        return
    
    def get_camera_pos(self):
        cam_state_pos = self._viewer.camera.pos
        cam_pos = np.array([float(cam_state_pos[0] - self._camera_offsets[0, 0]), 
                            float(cam_state_pos[1] - self._camera_offsets[0, 1]), 
                            float(cam_state_pos[2] - self._camera_offsets[0, 2])])
        return cam_pos
    
    def get_camera_dir(self):
        pitch = float(np.deg2rad(self._viewer.camera.pitch))
        yaw = float(np.deg2rad(self._viewer.camera.yaw))

        dx = np.cos(pitch) * np.cos(yaw)
        dy = np.cos(pitch) * np.sin(yaw)
        dz = np.sin(pitch)

        cam_dir = np.array([dx, dy, dz])
        cam_dir /= np.linalg.norm(cam_dir)
        return cam_dir
    
    def render(self):
        time_step = self.get_timestep()
        sim_time = time_step * self._sim_step_count

        self._viewer.end_frame()
        self._viewer.begin_frame(sim_time)
        self._viewer.log_state(self._sim_state.raw_state)

        self._draw_line_count = 0
        return
    
    def get_timestep(self):
        return self._timestep
    
    def get_sim_timestep(self):
        return self._sim_timestep
    
    def get_num_envs(self):
        return self._num_envs
    
    def get_objs_per_env(self):
        return len(self._sim_state.root_pos)
    
    def get_root_pos(self, obj_id):
        root_pos = self._sim_state.root_pos[obj_id]
        return root_pos
    
    def get_root_rot(self, obj_id):
        root_rot = self._sim_state.root_rot[obj_id]
        return root_rot
    
    def get_root_vel(self, obj_id):
        root_vel = self._sim_state.root_vel[obj_id]
        return root_vel
    
    def get_root_ang_vel(self, obj_id):
        root_ang_vel = self._sim_state.root_ang_vel[obj_id]
        return root_ang_vel
    
    def get_dof_pos(self, obj_id):
        dof_pos = self._sim_state.dof_pos[obj_id]
        return dof_pos
    
    def get_dof_vel(self, obj_id):
        dof_vel = self._sim_state.dof_vel[obj_id]
        return dof_vel
    
    def get_dof_forces(self, obj_id):
        dof_forces = self._dof_forces[obj_id]
        if (self._control_mode == engine.ControlMode.pos):
            dof_forces = torch.sum(dof_forces, dim=-1)
        return dof_forces
    
    def get_body_pos(self, obj_id):
        body_pos = self._sim_state.body_pos[obj_id]
        return body_pos
    
    def get_body_rot(self, obj_id):
        body_rot = self._sim_state.body_rot[obj_id]
        return body_rot
    
    def get_body_vel(self, obj_id):
        body_vel = self._sim_state.body_vel[obj_id]
        return body_vel
    
    def get_body_ang_vel(self, obj_id):
        body_ang_vel = self._sim_state.body_ang_vel[obj_id]
        return body_ang_vel
    
    def get_contact_forces(self, obj_id):
        forces = self._contact_forces[obj_id]
        return forces
    
    def get_ground_contact_forces(self, obj_id):
        forces = self._ground_contact_forces[obj_id]
        return forces
    
    def set_root_pos(self, env_id, obj_id, root_pos):
        if (env_id is None):
            self._sim_state.root_pos[obj_id][:, :] = root_pos
        else:
            self._sim_state.root_pos[obj_id][env_id, :] = root_pos
        return
    
    def set_root_rot(self, env_id, obj_id, root_rot):
        if (env_id is None):
            self._sim_state.root_rot[obj_id][:, :] = root_rot
        else:
            self._sim_state.root_rot[obj_id][env_id, :] = root_rot
        return
    
    def set_root_vel(self, env_id, obj_id, root_vel):
        if (env_id is None):
            self._sim_state.root_vel[obj_id][:, :] = root_vel
        else:
            self._sim_state.root_vel[obj_id][env_id, :] = root_vel
        return
    
    def set_root_ang_vel(self, env_id, obj_id, root_ang_vel):
        if (env_id is None):
            self._sim_state.root_ang_vel[obj_id][:, :] = root_ang_vel
        else:
            self._sim_state.root_ang_vel[obj_id][env_id, :] = root_ang_vel
        return
    
    def set_dof_pos(self, env_id, obj_id, dof_pos):
        if (env_id is None):
            self._sim_state.dof_pos[obj_id][:, :] = dof_pos
        else:
            self._sim_state.dof_pos[obj_id][env_id, :] = dof_pos
        return
    
    def set_dof_vel(self, env_id, obj_id, dof_vel):
        if (env_id is None):
            self._sim_state.dof_vel[obj_id][:, :] = dof_vel
        else:
            self._sim_state.dof_vel[obj_id][env_id, :] = dof_vel
        return
    
    def set_body_pos(self, env_id, obj_id, body_pos):
        if (env_id is None):
            self._sim_state.body_pos[obj_id][:, :, :] = body_pos
        else:
            self._sim_state.body_pos[obj_id][env_id, :, :] = body_pos
        return
    
    def set_body_rot(self, env_id, obj_id, body_rot):
        if (env_id is None):
            self._sim_state.body_rot[obj_id][:, :, :] = body_rot
        else:
            self._sim_state.body_rot[obj_id][env_id, :, :] = body_rot
        return
    
    def set_body_vel(self, env_id, obj_id, body_vel):
        if (env_id is None):
            self._sim_state.body_vel[obj_id][:, :, :] = body_vel
        else:
            self._sim_state.body_vel[obj_id][env_id, :, :] = body_vel
        return
    
    def set_body_ang_vel(self, env_id, obj_id, body_ang_vel):
        if (env_id is None):
            self._sim_state.body_ang_vel[obj_id][:, :, :] = body_ang_vel
        else:
            self._sim_state.body_ang_vel[obj_id][env_id, :, :] = body_ang_vel
        return
    
    def set_body_forces(self, env_id, obj_id, body_id, forces):
        if (env_id is None or len(env_id) > 0):
            assert(len(forces.shape) == 2)
            self._has_body_forces.fill_(1)

            if (env_id is None):
                self._sim_state.body_force[obj_id][:, body_id, :3] = forces
            else:
                self._sim_state.body_force[obj_id][env_id, body_id, :3] = forces
        return
    
    def get_obj_torque_limits(self, env_id, obj_id):
        objs_per_env = self.get_objs_per_env()
        obj_idx = env_id * objs_per_env + obj_id

        articulation_start = self._sim_model.articulation_start.numpy()
        joint_qd_start = self._sim_model.joint_qd_start.numpy()
        effort_lim = self._sim_model.joint_effort_limit.numpy()

        body_start = articulation_start[obj_idx]
        body_end = articulation_start[obj_idx + 1]
        qd_start = joint_qd_start[body_start + 1]
        qd_end = joint_qd_start[body_end]
        
        torque_lim = effort_lim[qd_start:qd_end]
        return torque_lim
    
    def get_obj_dof_limits(self, env_id, obj_id):
        objs_per_env = self.get_objs_per_env()
        obj_idx = env_id * objs_per_env + obj_id

        articulation_start = self._sim_model.articulation_start.numpy()
        joint_qd_start = self._sim_model.joint_qd_start.numpy()
        lim_low = self._sim_model.joint_limit_lower.numpy()
        lim_upper = self._sim_model.joint_limit_upper.numpy()

        body_start = articulation_start[obj_idx]
        body_end = articulation_start[obj_idx + 1]
        qd_start = joint_qd_start[body_start + 1]
        qd_end = joint_qd_start[body_end]
        
        dof_low = lim_low[qd_start:qd_end]
        dof_high = lim_upper[qd_start:qd_end]

        return dof_low, dof_high
    
    def find_obj_body_id(self, obj_id, body_name):
        body_names = self.get_obj_body_names(obj_id)
        body_id = body_names.index(body_name)
        return  body_id
    
    def get_obj_type(self, obj_id):
        obj_type = self._obj_types[0][obj_id]
        return obj_type
    
    def get_obj_num_bodies(self, obj_id):
        num_bodies = self._sim_state.body_pos[obj_id].shape[-2]
        return num_bodies
    
    def get_obj_num_dofs(self, obj_id):
        num_dofs = self._sim_state.dof_pos[obj_id].shape[-1]
        return num_dofs
    
    def get_obj_body_names(self, obj_id):
        articulation_start = self._sim_model.articulation_start.numpy()
        body_start = articulation_start[obj_id]
        body_end = articulation_start[obj_id + 1]
        body_names = self._sim_model.body_key[body_start:body_end]
        return body_names
    
    def calc_obj_mass(self, env_id, obj_id):
        objs_per_env = self.get_objs_per_env()
        obj_idx = env_id * objs_per_env + obj_id
        
        articulation_start = self._sim_model.articulation_start.numpy()
        body_masses = self._sim_model.body_mass.numpy()

        body_start = int(articulation_start[obj_idx])
        body_end = int(articulation_start[obj_idx + 1])
        masses = body_masses[body_start:body_end]
        total_mass = float(masses.sum())

        return total_mass
    
    def get_control_mode(self):
        return self._control_mode
    
    def set_body_color(self, env_id, obj_id, body_id, color):
        objs_per_env = self.get_objs_per_env()
        articulation_start = self._sim_model.articulation_start.numpy()

        obj_idx = env_id * objs_per_env + obj_id
        body_start = articulation_start[obj_idx]
        body_idx = body_start + body_id

        body_shape_idx = self._sim_model.body_shapes[body_idx]
        col_dict = dict([(i, color) for i in body_shape_idx])
        self._viewer.update_shape_colors(col_dict)
        return
    
    def draw_lines(self, env_id, start_verts, end_verts, cols, line_width):
        cam_offset = self._camera_offsets[env_id]
        start_pts = start_verts.copy()
        end_pts = end_verts.copy()

        start_pts[:, :2] += cam_offset[:2]
        end_pts[:, :2] += cam_offset[:2]
        line_name = "lines{:d}".format(self._draw_line_count)
        self._viewer.log_lines(name=line_name, 
                               starts=wp.array(start_pts, dtype=wp.vec3), 
                               ends=wp.array(end_pts, dtype=wp.vec3), 
                               colors=wp.array(cols[:, :3], dtype=wp.vec3), 
                               width=line_width)
        self._draw_line_count += 1
        return
    
    def register_keyboard_callback(self, key_str, callback_func):
        key_code = str_to_key_code(key_str)
        assert(key_code not in self._keyboard_callbacks)
        self._keyboard_callbacks[key_code] = callback_func
        return
    
    def _validate_envs(self):
        num_envs = self.get_num_envs()
        objs_per_env = len(self._obj_builders[0])

        for i in range(0, num_envs):
            curr_objs = self._obj_builders[i]
            num_objs = len(curr_objs)
            assert(num_objs == objs_per_env), "All envs must have the same number of objects."
        return

    def _build_objs(self):
        num_envs = self.get_num_envs()
        objs_per_env = len(self._obj_builders[0])
        total_objs = num_envs * objs_per_env

        for env_id in range(num_envs):
            self._scene_builder.begin_world()

            for obj_id in range(objs_per_env):
                obj_builder = self._obj_builders[env_id][obj_id]
                self._scene_builder.add_builder(obj_builder)
                
                num_created = env_id * objs_per_env + obj_id + 1
                Logger.print("Building {:d}/{:d} objs".format(num_created, total_objs), end='\r')

            self._scene_builder.end_world()

        Logger.print("")
        return
    
    def _apply_start_xform(self):
        num_envs = self.get_num_envs()
        objs_per_env = self.get_objs_per_env()
        for env_id in range(num_envs):
            for obj_id in range(objs_per_env):
                start_pos = self._obj_start_pos[env_id][obj_id]
                start_rot = self._obj_start_rot[env_id][obj_id]
                start_pos = torch.tensor(start_pos, device=self._device)
                start_rot = torch.tensor(start_rot, device=self._device)

                self.set_root_pos(env_id, obj_id, start_pos)
                self.set_root_rot(env_id, obj_id, start_rot)
        
        self._sim_state.eval_fk()
        return
    
    def _build_controls(self):
        kp = self._sim_model.mujoco.dof_passive_stiffness
        kd = self._sim_model.mujoco.dof_passive_damping

        if (self._control_mode == engine.ControlMode.none):
            self._sim_model.joint_target_ke.fill_(0.0)
            self._sim_model.joint_target_kd.fill_(0.0)

        elif (self._control_mode == engine.ControlMode.pos):
            wp.copy(self._sim_model.joint_target_ke, kp)
            wp.copy(self._sim_model.joint_target_kd, kd)

        elif (self._control_mode == engine.ControlMode.vel):
            self._sim_model.joint_target_ke.fill_(0.0)
            wp.copy(self._sim_model.joint_target_kd, kd)

        elif (self._control_mode == engine.ControlMode.torque):
            self._sim_model.joint_target_ke.fill_(0.0)
            self._sim_model.joint_target_kd.fill_(0.0)

        elif (self._control_mode == engine.ControlMode.pd_explicit):
            self._sim_model.joint_target_ke.fill_(0.0)
            self._sim_model.joint_target_kd.fill_(0.0)
            self._kp_raw = wp.clone(kp)
            self._kd_raw = wp.clone(kd)
            self._torque_lim_raw = wp.clone(self._sim_model.joint_effort_limit)

        else:
            assert(False), "Unsupported control mode: {}".format(self._control_mode)
        
        self._sim_model.mujoco.dof_passive_stiffness.fill_(0.0)
        self._sim_model.mujoco.dof_passive_damping.fill_(0.0)

        num_envs = self.get_num_envs()
        self._controls = Controls(self._sim_model, num_envs)
        return
    
    def _build_solver(self):
        self._solver = newton.solvers.SolverMuJoCo(
            self._sim_model,
            solver="newton",
            njmax=450,
            nconmax=150,
            impratio=10,
            iterations=100,
            ls_iterations=50
        )
        return
    
    def _build_graphs(self):
        self._graph = None
        
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self._simulate()

            self._graph = capture.graph
        return
    
    def _build_sim_tensors(self):
        self._raw_dof_pos = torch.zeros_like(self._sim_state._torch_joint_q)
        self._obj_dof_pos = []
        self._has_body_forces = wp.array([0], dtype=wp.int32)
        
        articulation_start = self._sim_model.articulation_start.numpy()
        joint_q_start = self._sim_model.joint_q_start.numpy()

        num_envs = self.get_num_envs()
        num_objs = self._sim_model.articulation_count
        objs_per_env = int(num_objs / num_envs)
        dof_pos = self._raw_dof_pos.view(num_envs, -1)

        for obj_id in range(objs_per_env):
            body_start = articulation_start[obj_id]
            body_end = articulation_start[obj_id + 1]
            q_start = joint_q_start[body_start + 1]
            q_end = joint_q_start[body_end]

            obj_dof_pos = dof_pos[:, q_start:q_end]
            self._obj_dof_pos.append(obj_dof_pos)
        return
    
    def _build_dof_force_tensors(self):
        if (self._control_mode == engine.ControlMode.none):
            self._dof_forces = self._controls.joint_force

        elif (self._control_mode == engine.ControlMode.pos):
            actuator_force = wp.to_torch(self._solver.mjw_data.actuator_force)
            self._dof_forces = actuator_force.view(actuator_force.shape[0], -1, 2)

        elif (self._control_mode == engine.ControlMode.vel):
            actuator_force = wp.to_torch(self._solver.mjw_data.actuator_force)
            actuator_force = actuator_force.view(actuator_force.shape[0], -1, 2)
            self._dof_forces = actuator_force[:, :, 1]

        elif (self._control_mode == engine.ControlMode.torque):
            self._dof_forces = self._controls.joint_force

        elif (self._control_mode == engine.ControlMode.pd_explicit):
            self._dof_forces = self._controls.joint_force

        else:
            assert(False), "Unsupported control mode: {}".format(self._control_mode)
        return
    
    def _create_obj_builder(self, asset_file, fix_root, is_visual, enable_self_collisions):
        obj_cfg = ObjCfg(asset_file=asset_file, fix_root=fix_root, is_visual=is_visual, 
                         enable_self_collisions=enable_self_collisions)
        
        if (obj_cfg in self._builder_cache):
            obj_builder = self._builder_cache[obj_cfg]
        else:
            obj_builder = self._create_model_builder()
            
            _, asset_ext = os.path.splitext(asset_file)
            if (asset_ext == ".xml"):
                obj_builder.add_mjcf(
                    asset_file,
                    floating=not fix_root,
                    collapse_fixed_joints=False,
                    enable_self_collisions=enable_self_collisions,
                    convert_3d_hinge_to_ball_joints=True
                )
            elif (asset_ext == ".urdf"):
                obj_builder.add_urdf(
                    asset_file,
                    floating=not fix_root,
                    collapse_fixed_joints=False,
                    enable_self_collisions=enable_self_collisions,
                    joint_ordering="dfs"
                )
            else:
                assert(False), "Unsupported asset format: {:s}".format(asset_ext)

            if (is_visual):
                for i in range(len(obj_builder.shape_key)):
                    obj_builder.shape_flags[i] &= ~newton.ShapeFlags.COLLIDE_SHAPES
            
            self._builder_cache[obj_cfg] = obj_builder

        return obj_builder
    
    def _create_model_builder(self):
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg.mu = 1.0
        return builder
    
    def _build_viewer(self):
        self._viewer = newton.viewer.ViewerGL(headless=False)
        self._draw_line_count = 0

        def on_keyboard_event(symbol, modifiers):
            self._on_keyboard_event(symbol, modifiers)
            return
        
        self._viewer.renderer.register_key_press(on_keyboard_event)
        return

    def _build_ground(self):
        shape_cfg = self._scene_builder.ShapeConfig(mu=1.0, restitution=0)
        self._scene_builder.add_ground_plane(cfg=shape_cfg)
        return
    
    def _build_contact_sensors(self):
        self._build_ground_contact_sensor()
        return

    def _build_ground_contact_sensor(self):
        self._ground_contact_sensor = newton.sensors.SensorContact(self._sim_model,
                                                                   sensing_obj_bodies="*",
                                                                   counterpart_shapes="ground*",
                                                                   include_total=True,
                                                                   verbose=True)
        num_envs = self.get_num_envs()
        num_objs = self.get_objs_per_env()
        ground_forces = wp.to_torch(self._ground_contact_sensor.net_force)
        ground_forces = ground_forces.view(num_envs, -1, 2, 3)

        offset_idx = 0
        self._contact_forces = []
        self._ground_contact_forces = []

        for obj_id in range(num_objs):
            num_bodies = self.get_obj_num_bodies(obj_id)
            obj_total_forces = ground_forces[:, offset_idx:offset_idx + num_bodies, 0, :]
            obj_ground_forces = ground_forces[:, offset_idx:offset_idx + num_bodies, 1, :]

            self._contact_forces.append(obj_total_forces)
            self._ground_contact_forces.append(obj_ground_forces)
            offset_idx += num_bodies
        return
    
    def _init_rendering(self):
        env_spacing = self._get_env_spacing()
        self._viewer.set_model(self._sim_model)
        self._viewer.set_world_offsets([env_spacing, env_spacing, 0.0])
        self._camera_offsets = self._viewer.world_offsets.numpy()

        self._apply_obj_colors()
        return
    
    def _get_env_spacing(self):
        return self._env_spacing
    
    def _simulate(self):
        state0 = self._sim_state.raw_state
        state1 = self._state_swap_buffer
        control = self._controls.control

        self._sim_state.pre_step_update()

        for i in range(self._sim_steps):
            self._pre_sim_step(state0, control)
            self._solver.step(state0, state1, control, self._contacts, self._sim_timestep)
            state0, state1 = state1, state0

        if (self._sim_steps % 2 != 0):
            self._sim_state.copy(self._state_swap_buffer)
        
        self._sim_state.post_step_update()
        
        def clear_body_forces():
            self._sim_state.clear_forces()
            self._has_body_forces.zero_()
            return
        wp.capture_if(self._has_body_forces, on_true=clear_body_forces)

        if (not isinstance(self._solver, newton.solvers.SolverMuJoCo) 
            and not isinstance(self._solver, newton.solvers.SolverFeatherstone)):
            newton.eval_ik(self._sim_model, state0, state0.joint_q, state0.joint_qd)
        return
    
    def _pre_sim_step(self, raw_state, control):
        raw_state.clear_forces()

        def apply_body_force():
            wp.copy(raw_state.body_f, self._sim_state._wp_body_force)
            return
        wp.capture_if(self._has_body_forces, on_true=apply_body_force)
        
        if (self._visualize()):
            self._viewer.apply_forces(raw_state)

        if (self._control_mode == engine.ControlMode.pd_explicit):
            self._apply_pd_explicit_torque(raw_state, control)
        return
    
    def _update_contact_sensors(self):
        newton.sensors.populate_contacts(self._contacts, self._solver)
        self._ground_contact_sensor.eval(self._contacts)
        return
    
    def _visualize(self):
        return self._viewer is not None
    
    def _apply_obj_colors(self):
        num_envs = self.get_num_envs()
        objs_per_env = self.get_objs_per_env()

        for env_id in range(num_envs):
            for obj_id in range(objs_per_env):
                obj_col = self._obj_colors[env_id][obj_id]

                if (obj_col is not None):
                    num_bodies = self.get_obj_num_bodies(obj_id)
                    for body_id in range(num_bodies):
                        self.set_body_color(env_id, obj_id, body_id, obj_col)
        return
    
    def _apply_pd_explicit_torque(self, sim_state, control):
        dof_pos = sim_state.joint_q
        dof_vel = sim_state.joint_qd
        tar_dof = control.joint_target_pos

        torque = self._kp_raw * (tar_dof - dof_pos) - self._kd_raw * dof_vel
        wp.launch(
            kernel=clamp_arrays,
            dim=torque.shape[0],
            inputs=[torque, -self._torque_lim_raw, self._torque_lim_raw]
        )
        wp.copy(control.joint_f, torque)
        return
    
    def _on_keyboard_event(self, symbol, modifiers):
        if (symbol in self._keyboard_callbacks):
            callback = self._keyboard_callbacks[symbol]
            callback()
        return


@wp.kernel
def clamp_arrays(x: wp.array(dtype=float),
                 mn: wp.array(dtype=float),
                 mx: wp.array(dtype=float)):
    i = wp.tid()
    x[i] = wp.clamp(x[i], mn[i], mx[i])
    return

@wp.kernel
def copy_indexed(input: wp.array(dtype=float),
                 idx_in: wp.array(dtype=int),
                 idx_out: wp.array(dtype=int),
                 output: wp.array(dtype=float)):
    tid = wp.tid()
    value = input[idx_in[tid]]
    output[idx_out[tid]] = value
    return

@wp.kernel
def quat_to_exp_map_indexed(in_q: wp.array(dtype=float),
                 q_start: wp.array(dtype=int),
                 dof_start: wp.array(dtype=int),
                 out_dof: wp.array(dtype=float)):
    tid = wp.tid()
    q_idx = q_start[tid]
    dof_idx = dof_start[tid]

    q = wp.quaternion(in_q[q_idx], in_q[q_idx + 1], in_q[q_idx + 2], in_q[q_idx + 3])
    axis, angle = wp.quat_to_axis_angle(q)
    exp_map = axis * angle

    out_dof[dof_idx] = exp_map[0]
    out_dof[dof_idx + 1] = exp_map[1]
    out_dof[dof_idx + 2] = exp_map[2]
    return

@wp.kernel
def exp_map_to_quat_indexed(in_dof: wp.array(dtype=float),
                 dof_start: wp.array(dtype=int),
                 q_start: wp.array(dtype=int),
                 out_q: wp.array(dtype=float)):
    min_theta = 1e-5

    tid = wp.tid()
    dof_idx = dof_start[tid]
    q_idx = q_start[tid]

    exp_map = wp.vec3f(in_dof[dof_idx], in_dof[dof_idx + 1], in_dof[dof_idx + 2])

    angle = wp.length(exp_map)
    axis = exp_map / angle
    angle = wp.atan2(wp.sin(angle), wp.cos(angle))

    mask = wp.abs(angle) > min_theta
    angle = angle if mask else 0.0
    axis[0] = axis[0] if mask else 0.0
    axis[1] = axis[1] if mask else 0.0
    axis[2] = axis[2] if mask else 1.0

    q = wp.quat_from_axis_angle(axis, angle)

    out_q[q_idx] = q[0]
    out_q[q_idx + 1] = q[1]
    out_q[q_idx + 2] = q[2]
    out_q[q_idx + 3] = q[3]
    return