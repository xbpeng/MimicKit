"""Warp/Newton engine using XPBD solver and ViewerGL.

XPBD (Extended Position-Based Dynamics) is the contact resolution algorithm
used in NVIDIA PhysX 5, matching Isaac Lab's physics substrate.

Physics settings target parity with Isaac Lab:
  - friction=1.0  (matches Isaac Lab static/dynamic friction)
  - xpbd_iterations=4  (matches Isaac Lab max_position_iteration_count)
  - sim_freq=120 Hz, control_freq=30 Hz
  - PD gains loaded directly from MJCF stiffness/damping attributes
"""

import warp as wp
wp.config.enable_backward = False

import newton
import numpy as np
import os
import torch
import pyglet

import engines.engine as engine
from engines.newton_engine import (
    SimState, Controls, ObjCfg,
    str_to_key_code, clamp_arrays,
    copy_indexed, quat_to_exp_map_indexed, exp_map_to_quat_indexed,
)
from util.logger import Logger


class WarpEngine(engine.Engine):
    def __init__(self, config, num_envs, device, visualize, record_video=False):
        super().__init__(visualize=visualize)

        self._device = device
        self._num_envs = num_envs
        self._env_spacing = config["env_spacing"]

        sim_freq = config.get("sim_freq", 120)
        control_freq = config.get("control_freq", 30)
        assert sim_freq >= control_freq and sim_freq % control_freq == 0, \
            "sim_freq must be a multiple of control_freq"

        self._timestep = 1.0 / control_freq
        self._sim_steps = int(sim_freq / control_freq)
        self._sim_timestep = 1.0 / sim_freq
        self._sim_step_count = 0
        self._xpbd_iterations = config.get("xpbd_iterations", 4)

        self._scene_builder = self._create_model_builder()
        self._builder_cache = dict()

        self._obj_types = []
        self._obj_builders = []
        self._obj_start_pos = []
        self._obj_start_rot = []
        self._obj_colors = []

        if "control_mode" in config:
            self._control_mode = engine.ControlMode[config["control_mode"]]
        else:
            self._control_mode = engine.ControlMode.none

        self._build_ground()

        if visualize:
            self._build_viewer()
            self._keyboard_callbacks = dict()
        else:
            self._viewer = None

        return

    def get_name(self):
        return "warp"

    def create_env(self):
        env_id = len(self._obj_builders)
        assert env_id < self.get_num_envs()
        self._obj_builders.append([])
        self._obj_types.append([])
        self._obj_start_pos.append([])
        self._obj_start_rot.append([])
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

        self._apply_start_xform()
        self._needs_fk = False  # eval_fk already ran in _apply_start_xform
        self._build_controls()
        self._build_contact_sensors()
        self._build_solver()
        self._build_dof_force_tensors()

        self._contacts = self._sim_model.contacts()

        if self._visualize():
            self._init_viewer()
            self._camera_offsets = self._viewer.world_offsets.numpy()

        self._build_graphs()
        return

    def step(self):
        if self._graph:
            wp.capture_launch(self._graph)
        else:
            self._simulate()

        self._ground_contact_sensor.update(self._contacts)
        self._sim_step_count += 1
        return

    def create_obj(self, env_id, obj_type, asset_file, name, is_visual=False,
                   enable_self_collisions=True, fix_root=False, start_pos=None,
                   start_rot=None, color=None, disable_motors=False):
        if start_rot is None:
            start_rot = np.array([0.0, 0.0, 0.0, 1.0])
        if start_pos is None:
            start_pos = np.array([0.0, 0.0, 0.0])

        obj_builder = self._create_obj_builder(asset_file=asset_file, fix_root=fix_root,
                                               is_visual=is_visual,
                                               enable_self_collisions=enable_self_collisions)
        obj_id = len(self._obj_builders[env_id])
        self._obj_types[env_id].append(obj_type)
        self._obj_builders[env_id].append(obj_builder)
        self._obj_start_pos[env_id].append(start_pos)
        self._obj_start_rot[env_id].append(start_rot)
        self._obj_colors[env_id].append(color)
        return obj_id

    def set_cmd(self, obj_id, cmd):
        if self._control_mode == engine.ControlMode.none:
            pass
        elif self._control_mode == engine.ControlMode.pos:
            self._controls.target_pos[obj_id][:] = cmd
        elif self._control_mode == engine.ControlMode.vel:
            self._controls.target_vel[obj_id][:] = cmd
        elif self._control_mode == engine.ControlMode.torque:
            self._controls.joint_force[obj_id][:] = cmd
        elif self._control_mode == engine.ControlMode.pd_explicit:
            self._controls.target_pos[obj_id][:] = cmd
        else:
            assert False, "Unsupported control mode: {}".format(self._control_mode)
        return

    def set_camera_pose(self, pos, look_at):
        dx = look_at[0] - pos[0]
        dy = look_at[1] - pos[1]
        dz = look_at[2] - pos[2]
        pitch = np.arctan2(dz, np.sqrt(dx * dx + dy * dy))
        yaw = np.arctan2(dy, dx)
        cam_pos = pyglet.math.Vec3(
            pos[0] + self._camera_offsets[0, 0],
            pos[1] + self._camera_offsets[0, 1],
            pos[2] + self._camera_offsets[0, 2],
        )
        self._viewer.set_camera(cam_pos, np.rad2deg(pitch), np.rad2deg(yaw))
        return

    def get_camera_pos(self):
        p = self._viewer.camera.pos
        return np.array([
            float(p[0] - self._camera_offsets[0, 0]),
            float(p[1] - self._camera_offsets[0, 1]),
            float(p[2] - self._camera_offsets[0, 2]),
        ])

    def get_camera_dir(self):
        pitch = float(np.deg2rad(self._viewer.camera.pitch))
        yaw   = float(np.deg2rad(self._viewer.camera.yaw))
        d = np.array([np.cos(pitch) * np.cos(yaw),
                      np.cos(pitch) * np.sin(yaw),
                      np.sin(pitch)])
        return d / np.linalg.norm(d)

    def render(self):
        sim_time = self._timestep * self._sim_step_count
        self._viewer.end_frame()
        self._viewer.begin_frame(sim_time)
        self._viewer.log_state(self._sim_state.raw_state)
        self._draw_line_count = 0
        super().render()
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
        return self._sim_state.root_pos[obj_id]

    def get_root_rot(self, obj_id):
        return self._sim_state.root_rot[obj_id]

    def get_root_vel(self, obj_id):
        return self._sim_state.root_vel[obj_id]

    def get_root_ang_vel(self, obj_id):
        return self._sim_state.root_ang_vel[obj_id]

    def get_dof_pos(self, obj_id):
        return self._sim_state.dof_pos[obj_id]

    def get_dof_vel(self, obj_id):
        return self._sim_state.dof_vel[obj_id]

    def get_dof_forces(self, obj_id):
        return self._dof_forces[obj_id]

    def get_body_pos(self, obj_id):
        return self._sim_state.body_pos[obj_id]

    def get_body_rot(self, obj_id):
        return self._sim_state.body_rot[obj_id]

    def get_body_vel(self, obj_id):
        return self._sim_state.body_vel[obj_id]

    def get_body_ang_vel(self, obj_id):
        return self._sim_state.body_ang_vel[obj_id]

    def get_contact_forces(self, obj_id):
        return self._contact_forces[obj_id]

    def get_ground_contact_forces(self, obj_id):
        return self._ground_contact_forces[obj_id]

    def set_root_pos(self, env_id, obj_id, root_pos):
        if env_id is None:
            self._sim_state.root_pos[obj_id][:, :] = root_pos
        else:
            self._sim_state.root_pos[obj_id][env_id, :] = root_pos
        self._needs_fk = True

    def set_root_rot(self, env_id, obj_id, root_rot):
        if env_id is None:
            self._sim_state.root_rot[obj_id][:, :] = root_rot
        else:
            self._sim_state.root_rot[obj_id][env_id, :] = root_rot
        self._needs_fk = True

    def set_root_vel(self, env_id, obj_id, root_vel):
        if env_id is None:
            self._sim_state.root_vel[obj_id][:, :] = root_vel
        else:
            self._sim_state.root_vel[obj_id][env_id, :] = root_vel
        self._needs_fk = True

    def set_root_ang_vel(self, env_id, obj_id, root_ang_vel):
        if env_id is None:
            self._sim_state.root_ang_vel[obj_id][:, :] = root_ang_vel
        else:
            self._sim_state.root_ang_vel[obj_id][env_id, :] = root_ang_vel
        self._needs_fk = True

    def set_dof_pos(self, env_id, obj_id, dof_pos):
        if env_id is None:
            self._sim_state.dof_pos[obj_id][:, :] = dof_pos
        else:
            self._sim_state.dof_pos[obj_id][env_id, :] = dof_pos
        self._needs_fk = True

    def set_dof_vel(self, env_id, obj_id, dof_vel):
        if env_id is None:
            self._sim_state.dof_vel[obj_id][:, :] = dof_vel
        else:
            self._sim_state.dof_vel[obj_id][env_id, :] = dof_vel
        self._needs_fk = True

    def set_body_pos(self, env_id, obj_id, body_pos):
        if env_id is None:
            self._sim_state.body_pos[obj_id][:, :, :] = body_pos
        else:
            self._sim_state.body_pos[obj_id][env_id, :, :] = body_pos

    def set_body_rot(self, env_id, obj_id, body_rot):
        if env_id is None:
            self._sim_state.body_rot[obj_id][:, :, :] = body_rot
        else:
            self._sim_state.body_rot[obj_id][env_id, :, :] = body_rot

    def set_body_vel(self, env_id, obj_id, body_vel):
        if env_id is None:
            self._sim_state.body_vel[obj_id][:, :, :] = body_vel
        else:
            self._sim_state.body_vel[obj_id][env_id, :, :] = body_vel

    def set_body_ang_vel(self, env_id, obj_id, body_ang_vel):
        if env_id is None:
            self._sim_state.body_ang_vel[obj_id][:, :, :] = body_ang_vel
        else:
            self._sim_state.body_ang_vel[obj_id][env_id, :, :] = body_ang_vel

    def set_body_forces(self, env_id, obj_id, body_id, forces):
        if env_id is None or len(env_id) > 0:
            assert len(forces.shape) == 2
            self._has_body_forces.fill_(1)
            if env_id is None:
                self._sim_state.body_force[obj_id][:, body_id, :3] = forces
            else:
                self._sim_state.body_force[obj_id][env_id, body_id, :3] = forces

    def get_obj_torque_limits(self, env_id, obj_id):
        obj_idx = env_id * self.get_objs_per_env() + obj_id
        art_start = self._sim_model.articulation_start.numpy()
        qd_start  = self._sim_model.joint_qd_start.numpy()
        eff_lim   = self._sim_model.joint_effort_limit.numpy()
        bs = art_start[obj_idx];  be = art_start[obj_idx + 1]
        return eff_lim[qd_start[bs + 1]:qd_start[be]]

    def get_obj_dof_limits(self, env_id, obj_id):
        obj_idx  = env_id * self.get_objs_per_env() + obj_id
        art_start = self._sim_model.articulation_start.numpy()
        qd_start  = self._sim_model.joint_qd_start.numpy()
        lim_lo    = self._sim_model.joint_limit_lower.numpy()
        lim_hi    = self._sim_model.joint_limit_upper.numpy()
        bs = art_start[obj_idx];  be = art_start[obj_idx + 1]
        s  = qd_start[bs + 1];   e  = qd_start[be]
        return lim_lo[s:e], lim_hi[s:e]

    def get_obj_pd_gains(self, env_id, obj_id):
        obj_idx   = env_id * self.get_objs_per_env() + obj_id
        art_start = self._sim_model.articulation_start.numpy()
        qd_start  = self._sim_model.joint_qd_start.numpy()
        bs = art_start[obj_idx];  be = art_start[obj_idx + 1]
        s  = qd_start[bs + 1];   e  = qd_start[be]
        kp = self._sim_model.joint_target_ke.numpy()[s:e]
        kd = self._sim_model.joint_target_kd.numpy()[s:e]
        return kp, kd

    def find_obj_body_id(self, obj_id, body_name):
        return self.get_obj_body_names(obj_id).index(body_name)

    def get_obj_type(self, obj_id):
        return self._obj_types[0][obj_id]

    def get_obj_num_bodies(self, obj_id):
        return self._sim_state.body_pos[obj_id].shape[-2]

    def get_obj_num_dofs(self, obj_id):
        return self._sim_state.dof_pos[obj_id].shape[-1]

    def get_obj_body_names(self, obj_id):
        art_start = self._sim_model.articulation_start.numpy()
        bs = art_start[obj_id];  be = art_start[obj_id + 1]
        labels = self._sim_model.body_label[bs:be]
        return [os.path.basename(l) for l in labels]

    def calc_obj_mass(self, env_id, obj_id):
        obj_idx   = env_id * self.get_objs_per_env() + obj_id
        art_start = self._sim_model.articulation_start.numpy()
        masses    = self._sim_model.body_mass.numpy()
        bs = int(art_start[obj_idx]);  be = int(art_start[obj_idx + 1])
        return float(masses[bs:be].sum())

    def get_control_mode(self):
        return self._control_mode

    def draw_lines(self, env_id, start_verts, end_verts, cols, line_width):
        cam_offset = self._camera_offsets[env_id]
        start_pts = start_verts.copy();  start_pts[:, :2] += cam_offset[:2]
        end_pts   = end_verts.copy();    end_pts[:, :2]   += cam_offset[:2]
        self._viewer.log_lines(
            name="lines{:d}".format(self._draw_line_count),
            starts=wp.array(start_pts, dtype=wp.vec3),
            ends=wp.array(end_pts,   dtype=wp.vec3),
            colors=wp.array(cols[:, :3], dtype=wp.vec3),
            width=line_width,
        )
        self._draw_line_count += 1

    def register_keyboard_callback(self, key_str, callback_func):
        key_code = str_to_key_code(key_str)
        assert key_code not in self._keyboard_callbacks
        self._keyboard_callbacks[key_code] = callback_func

    # ── Private ───────────────────────────────────────────────────────────────

    def _create_model_builder(self):
        builder = newton.ModelBuilder()
        # Register MuJoCo custom attributes so the MJCF loader populates
        # mujoco.dof_passive_stiffness/damping from joint stiffness/damping attrs.
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg.mu = 1.0  # match Isaac Lab friction
        return builder

    def _build_ground(self):
        shape_cfg = self._scene_builder.ShapeConfig(mu=1.0, restitution=0.0)
        self._scene_builder.add_ground_plane(cfg=shape_cfg)

    def _build_viewer(self):
        self._viewer = newton.viewer.ViewerGL(headless=False)
        self._draw_line_count = 0
        def on_key(symbol, modifiers):
            self._on_keyboard_event(symbol, modifiers)
        self._viewer.renderer.register_key_press(on_key)

    def _init_viewer(self):
        self._viewer.set_model(self._sim_model)
        self._viewer.set_world_offsets([self._env_spacing, self._env_spacing, 0.0])
        self._apply_obj_colors(self._viewer)

    def _validate_envs(self):
        n = len(self._obj_builders[0])
        for builders in self._obj_builders:
            assert len(builders) == n, "All envs must have the same number of objects."

    def _build_objs(self):
        num_envs = self.get_num_envs()
        objs_per_env = len(self._obj_builders[0])
        total = num_envs * objs_per_env
        for env_id in range(num_envs):
            self._scene_builder.begin_world()
            for obj_id in range(objs_per_env):
                self._scene_builder.add_builder(self._obj_builders[env_id][obj_id])
                Logger.print("Building {:d}/{:d} objs".format(
                    env_id * objs_per_env + obj_id + 1, total), end='\r')
            self._scene_builder.end_world()
        Logger.print("")

    def _apply_start_xform(self):
        for env_id in range(self.get_num_envs()):
            for obj_id in range(self.get_objs_per_env()):
                pos = torch.tensor(self._obj_start_pos[env_id][obj_id], device=self._device)
                rot = torch.tensor(self._obj_start_rot[env_id][obj_id], device=self._device)
                self.set_root_pos(env_id, obj_id, pos)
                self.set_root_rot(env_id, obj_id, rot)
        self._sim_state.eval_fk()

    def _build_controls(self):
        if self._control_mode == engine.ControlMode.none:
            self._sim_model.joint_target_ke.fill_(0.0)
            self._sim_model.joint_target_kd.fill_(0.0)
            self._sim_model.joint_target_mode.fill_(int(newton.JointTargetMode.NONE))

        elif self._control_mode == engine.ControlMode.pos:
            # Compute PD torques explicitly (matching Isaac Lab's force-based PD).
            # XPBD's internal compliance-based position control uses a different scaling
            # than PhysX's direct force application, so we bypass it entirely: compute
            # torques = kp*(target-pos) - kd*vel and feed them as EFFORT to XPBD.
            art_start = self._sim_model.articulation_start.numpy()
            qd_start  = self._sim_model.joint_qd_start.numpy()
            ke_all    = wp.to_torch(self._sim_model.mujoco.dof_passive_stiffness)
            kd_all    = wp.to_torch(self._sim_model.mujoco.dof_passive_damping)
            eff_all   = wp.to_torch(self._sim_model.joint_effort_limit)
            num_objs  = self._sim_model.articulation_count
            num_envs  = self.get_num_envs()
            objs_per_env = num_objs // num_envs

            self._kp_per_obj  = []
            self._kd_per_obj  = []
            self._lim_per_obj = []
            for obj_id in range(objs_per_env):
                bs = art_start[obj_id];  be = art_start[obj_id + 1]
                s  = qd_start[bs + 1];   e  = qd_start[be]
                self._kp_per_obj.append(ke_all[s:e].clone())
                self._kd_per_obj.append(kd_all[s:e].clone())
                self._lim_per_obj.append(eff_all[s:e].clone())

            # Use EFFORT mode so XPBD applies joint_f directly
            self._sim_model.joint_target_ke.fill_(0.0)
            self._sim_model.joint_target_kd.fill_(0.0)
            self._sim_model.joint_target_mode.fill_(int(newton.JointTargetMode.EFFORT))
            # Zero passive springs so they don't double-add stiffness
            self._sim_model.mujoco.dof_passive_stiffness.fill_(0.0)
            self._sim_model.mujoco.dof_passive_damping.fill_(0.0)

        elif self._control_mode == engine.ControlMode.vel:
            self._sim_model.joint_target_ke.fill_(0.0)
            self._sim_model.joint_target_mode.fill_(int(newton.JointTargetMode.VELOCITY))

        elif self._control_mode == engine.ControlMode.torque:
            self._sim_model.joint_target_ke.fill_(0.0)
            self._sim_model.joint_target_kd.fill_(0.0)
            self._sim_model.joint_target_mode.fill_(int(newton.JointTargetMode.EFFORT))

        else:
            assert False, "Unsupported control mode: {}".format(self._control_mode)

        self._controls = Controls(self._sim_model, self.get_num_envs())

    def _build_solver(self):
        self._solver = newton.solvers.SolverXPBD(
            self._sim_model,
            iterations=self._xpbd_iterations,
        )

    def _build_contact_sensors(self):
        self._ground_contact_sensor = newton.sensors.SensorContact(
            self._sim_model,
            sensing_obj_bodies="*",
            counterpart_shapes="ground*",
            include_total=True,
            verbose=True,
        )
        num_envs = self.get_num_envs()
        num_objs = self.get_objs_per_env()
        ground_forces = wp.to_torch(self._ground_contact_sensor.net_force)
        ground_forces = ground_forces.view(num_envs, -1, 2, 3)

        offset = 0
        self._contact_forces = []
        self._ground_contact_forces = []
        for obj_id in range(num_objs):
            nb = self.get_obj_num_bodies(obj_id)
            self._contact_forces.append(ground_forces[:, offset:offset + nb, 0, :])
            self._ground_contact_forces.append(ground_forces[:, offset:offset + nb, 1, :])
            offset += nb

    def _build_dof_force_tensors(self):
        # XPBD does not expose per-DOF actuator forces; expose joint_f as proxy.
        self._has_body_forces = wp.array([0], dtype=wp.int32)
        self._dof_forces = self._controls.joint_force

    def _build_graphs(self):
        # CUDA graph capture disabled: graph replay after episode resets causes
        # state inconsistencies with XPBD's maximal-coordinate representation.
        self._graph = None

    def _create_obj_builder(self, asset_file, fix_root, is_visual, enable_self_collisions):
        obj_cfg = ObjCfg(asset_file=asset_file, fix_root=fix_root,
                         is_visual=is_visual, enable_self_collisions=enable_self_collisions)
        if obj_cfg in self._builder_cache:
            return self._builder_cache[obj_cfg]

        obj_builder = self._create_model_builder()
        _, ext = os.path.splitext(asset_file)

        if ext == ".xml":
            obj_builder.add_mjcf(
                asset_file,
                floating=not fix_root,
                ignore_inertial_definitions=False,
                collapse_fixed_joints=False,
                enable_self_collisions=enable_self_collisions,
                convert_3d_hinge_to_ball_joints=False,  # keep as revolute joints to match Isaac Lab DOF layout
            )
        elif ext == ".urdf":
            obj_builder.add_urdf(
                asset_file,
                floating=not fix_root,
                ignore_inertial_definitions=False,
                collapse_fixed_joints=False,
                enable_self_collisions=enable_self_collisions,
                joint_ordering="dfs",
            )
        else:
            assert False, "Unsupported asset format: {:s}".format(ext)

        if is_visual:
            for i in range(len(obj_builder.shape_flags)):
                obj_builder.shape_flags[i] &= ~newton.ShapeFlags.COLLIDE_SHAPES

        self._builder_cache[obj_cfg] = obj_builder
        return obj_builder

    def _simulate(self):
        state0  = self._sim_state.raw_state
        state1  = self._state_swap_buffer
        control = self._controls.control

        # Only propagate joint_q → body_q when the env has written new state
        # (e.g. after a reset). Running eval_fk every step would overwrite the
        # valid XPBD maximal-coordinate body positions with stale joint coords.
        if self._needs_fk:
            self._sim_state.eval_fk()
            self._needs_fk = False
        self._sim_state.pre_step_update()

        for _ in range(self._sim_steps):
            self._pre_sim_step(state0, control)
            self._sim_model.collide(state0, self._contacts)  # required for XPBD
            self._solver.step(state0, state1, control, self._contacts, self._sim_timestep)
            state0, state1 = state1, state0

        if self._sim_steps % 2 != 0:
            self._sim_state.copy(self._state_swap_buffer)

        self._sim_state.post_step_update()

        # XPBD uses maximal coords; recompute generalised joint coords from body poses
        newton.eval_ik(self._sim_model, state0, state0.joint_q, state0.joint_qd)

        def clear_body_forces():
            self._sim_state.clear_forces()
            self._has_body_forces.zero_()
        wp.capture_if(self._has_body_forces, on_true=clear_body_forces)

    def _pre_sim_step(self, raw_state, control):
        raw_state.clear_forces()

        def apply_body_force():
            wp.copy(raw_state.body_f, self._sim_state._wp_body_force)
        wp.capture_if(self._has_body_forces, on_true=apply_body_force)

        if self._visualize():
            self._viewer.apply_forces(raw_state)

        if self._control_mode in (engine.ControlMode.pos, engine.ControlMode.pd_explicit):
            self._apply_pd_explicit_torque()

    def _apply_pd_explicit_torque(self):
        # Use SimState DOF views and Controls DOF views — both sliced to revolute
        # DOFs only, so sizes always match regardless of joint_q/joint_qd offset.
        # This exactly matches Isaac Lab's force-based PD: torque = kp*(target-pos) - kd*vel
        for obj_id in range(self.get_objs_per_env()):
            dof_pos = self._sim_state.dof_pos[obj_id]    # [num_envs, N]
            dof_vel = self._sim_state.dof_vel[obj_id]    # [num_envs, N]
            tar_pos = self._controls.target_pos[obj_id]  # [num_envs, N]
            kp      = self._kp_per_obj[obj_id]           # [N]
            kd      = self._kd_per_obj[obj_id]           # [N]
            lim     = self._lim_per_obj[obj_id]          # [N]

            torque = kp * (tar_pos - dof_pos) - kd * dof_vel
            torque = torch.clamp(torque, -lim, lim)
            self._controls.joint_force[obj_id][:] = torque

    def _apply_obj_colors(self, viewer):
        for env_id in range(self.get_num_envs()):
            for obj_id in range(self.get_objs_per_env()):
                col = self._obj_colors[env_id][obj_id]
                if col is not None:
                    for body_id in range(self.get_obj_num_bodies(obj_id)):
                        self._set_body_color(env_id, obj_id, body_id, col, viewer)

    def _set_body_color(self, env_id, obj_id, body_id, color, viewer):
        objs_per_env = self.get_objs_per_env()
        art_start = self._sim_model.articulation_start.numpy()
        obj_idx   = env_id * objs_per_env + obj_id
        body_idx  = art_start[obj_idx] + body_id
        col_dict  = {i: color for i in self._sim_model.body_shapes[body_idx]}
        viewer.update_shape_colors(col_dict)

    def _on_keyboard_event(self, symbol, modifiers):
        if symbol in self._keyboard_callbacks:
            self._keyboard_callbacks[symbol]()

    def _visualize(self):
        return self._viewer is not None
