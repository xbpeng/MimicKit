from __future__ import annotations

from dataclasses import dataclass
import math
import os

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp
from mujoco_warp._src import support as mjwarp_support

import engines.engine as engine
from util.logger import Logger
import util.torch_util as torch_util


wp.config.enable_backward = False

@dataclass(frozen=True)
class ObjDef:
    asset_file: str
    name: str
    obj_type: engine.ObjType
    is_visual: bool
    enable_self_collisions: bool
    fix_root: bool
    disable_motors: bool


@dataclass
class ObjMeta:
    obj_def: ObjDef
    body_ids: list[int]
    body_names: list[str]
    qvel_ids: list[int]
    qvel_col_ids: torch.Tensor
    actuator_ids: list[int]
    hinge_qpos_ids: list[int]
    hinge_dof_ids: list[int]
    hinge_qpos_col_ids: torch.Tensor
    hinge_local_col_ids: torch.Tensor
    ball_qpos_ids: list[int]
    ball_dof_ids: list[int]
    ball_local_starts: list[int]
    root_body_id: int
    root_qpos_ids: list[int]
    root_qvel_ids: list[int]
    dof_low: np.ndarray
    dof_high: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    torque_lim: np.ndarray
    mass: float


class SimState:
    def __init__(self, wp_data, obj_metas, num_envs, device, wp_device):
        self._wp_device = wp_device
        self._wp_data = wp_data

        num_objs = len(obj_metas)

        # MuJoCo stores each world in flat qpos/qvel/body arrays; expose
        # Newton-style per-object tensor views for the rest of MimicKit.
        self._torch_qpos = wp.to_torch(self._wp_data.qpos)
        self._torch_qvel = wp.to_torch(self._wp_data.qvel)
        self._torch_xfrc_applied = wp.to_torch(self._wp_data.xfrc_applied)
        self._torch_ctrl = wp.to_torch(self._wp_data.ctrl)

        total_bodies = sum(len(meta.body_ids) for meta in obj_metas)
        total_dofs = sum(len(meta.qvel_ids) for meta in obj_metas)

        self._num_envs = num_envs
        self._num_objs = num_objs
        self._num_bodies = total_bodies
        self._num_dofs = total_dofs

        self._wp_root_rot = wp.zeros((num_envs, num_objs, 4), device=self._wp_device, dtype=float)
        self._wp_root_vel = wp.zeros((num_envs, num_objs, 3), device=self._wp_device, dtype=float)
        self._wp_root_ang_vel = wp.zeros((num_envs, num_objs, 3), device=self._wp_device, dtype=float)

        self._torch_root_rot = wp.to_torch(self._wp_root_rot)
        self._torch_root_vel = wp.to_torch(self._wp_root_vel)
        self._torch_root_ang_vel = wp.to_torch(self._wp_root_ang_vel)

        self._wp_body_pos = wp.zeros((num_envs, total_bodies, 3), device=self._wp_device, dtype=float)
        self._wp_body_rot = wp.zeros((num_envs, total_bodies, 4), device=self._wp_device, dtype=float)
        self._wp_body_vel = wp.zeros((num_envs, total_bodies, 3), device=self._wp_device, dtype=float)
        self._wp_body_ang_vel = wp.zeros((num_envs, total_bodies, 3), device=self._wp_device, dtype=float)

        self._torch_body_pos = wp.to_torch(self._wp_body_pos)
        self._torch_body_rot = wp.to_torch(self._wp_body_rot)
        self._torch_body_vel = wp.to_torch(self._wp_body_vel)
        self._torch_body_ang_vel = wp.to_torch(self._wp_body_ang_vel)

        if (total_dofs > 0):
            self._wp_dof_pos = wp.zeros((num_envs, total_dofs), device=self._wp_device, dtype=float)
            self._wp_dof_vel = wp.zeros((num_envs, total_dofs), device=self._wp_device, dtype=float)
            self._torch_dof_pos = wp.to_torch(self._wp_dof_pos)
            self._torch_dof_vel = wp.to_torch(self._wp_dof_vel)
        else:
            self._wp_dof_pos = None
            self._wp_dof_vel = None

            self._torch_dof_pos = torch.zeros([num_envs, 0], device=device, dtype=torch.float32)
            self._torch_dof_vel = torch.zeros([num_envs, 0], device=device, dtype=torch.float32)

        root_qpos_start = []
        root_qvel_start = []
        root_body_id = []

        body_ids = []
        body_root_ids = []

        dof_qpos_start = []
        dof_qpos_kind = []
        dof_qpos_comp = []
        dof_qvel_id = []
        control_qvel_id = []
        kp = []
        kd = []
        torque_lim = []

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

        body_offset = 0
        dof_offset = 0

        for obj_id, meta in enumerate(obj_metas):
            num_bodies = len(meta.body_ids)
            num_dofs = len(meta.qvel_ids)

            if (len(meta.root_qpos_ids) > 0):
                root_qpos_start.append(meta.root_qpos_ids[0])
                root_qvel_start.append(meta.root_qvel_ids[0])
                obj_root_pos = self._torch_qpos[:, meta.root_qpos_ids[0]:meta.root_qpos_ids[0] + 3]
            else:
                root_qpos_start.append(-1)
                root_qvel_start.append(-1)
                obj_root_pos = self._torch_body_pos[:, body_offset, :]

            root_body_id.append(meta.root_body_id)

            for body_id in meta.body_ids:
                body_ids.append(body_id)
                body_root_ids.append(meta.root_body_id)

            qpos_by_qvel = {}
            for qpos_id, qvel_id in zip(meta.hinge_qpos_ids, meta.hinge_dof_ids):
                qpos_by_qvel[qvel_id] = (qpos_id, 0, 0)

            for qpos_start, qvel_start in zip(meta.ball_qpos_ids, meta.ball_dof_ids):
                for comp in range(3):
                    qpos_by_qvel[qvel_start + comp] = (qpos_start, 1, comp)

            for local_id, qvel_id in enumerate(meta.qvel_ids):
                qpos_start, qpos_kind, qpos_comp = qpos_by_qvel[qvel_id]
                dof_qpos_start.append(qpos_start)
                dof_qpos_kind.append(qpos_kind)
                dof_qpos_comp.append(qpos_comp)
                dof_qvel_id.append(qvel_id)
                control_qvel_id.append(-1 if meta.obj_def.disable_motors else qvel_id)
                kp.append(float(meta.kp[local_id]))
                kd.append(float(meta.kd[local_id]))
                torque_lim.append(float(meta.torque_lim[local_id]))

            self.root_pos.append(obj_root_pos)
            self.root_rot.append(self._torch_root_rot[:, obj_id, :])
            self.root_vel.append(self._torch_root_vel[:, obj_id, :])
            self.root_ang_vel.append(self._torch_root_ang_vel[:, obj_id, :])
            self.body_pos.append(self._torch_body_pos[:, body_offset:body_offset + num_bodies, :])
            self.body_rot.append(self._torch_body_rot[:, body_offset:body_offset + num_bodies, :])
            self.body_vel.append(self._torch_body_vel[:, body_offset:body_offset + num_bodies, :])
            self.body_ang_vel.append(self._torch_body_ang_vel[:, body_offset:body_offset + num_bodies, :])
            self.dof_pos.append(self._torch_dof_pos[:, dof_offset:dof_offset + num_dofs])
            self.dof_vel.append(self._torch_dof_vel[:, dof_offset:dof_offset + num_dofs])

            body_offset += num_bodies
            dof_offset += num_dofs

        self._wp_root_qpos_start = wp.array(root_qpos_start, device=self._wp_device, dtype=int)
        self._wp_root_qvel_start = wp.array(root_qvel_start, device=self._wp_device, dtype=int)
        self._wp_root_body_id = wp.array(root_body_id, device=self._wp_device, dtype=int)

        self._wp_body_ids = wp.array(body_ids, device=self._wp_device, dtype=int)
        self._wp_body_root_ids = wp.array(body_root_ids, device=self._wp_device, dtype=int)

        self._wp_dof_qpos_start = wp.array(dof_qpos_start, device=self._wp_device, dtype=int)
        self._wp_dof_qpos_kind = wp.array(dof_qpos_kind, device=self._wp_device, dtype=int)
        self._wp_dof_qpos_comp = wp.array(dof_qpos_comp, device=self._wp_device, dtype=int)
        self._wp_dof_qvel_id = wp.array(dof_qvel_id, device=self._wp_device, dtype=int)
        self._wp_control_qvel_id = wp.array(control_qvel_id, device=self._wp_device, dtype=int)
        self._wp_kp = wp.array(kp, device=self._wp_device, dtype=float)
        self._wp_kd = wp.array(kd, device=self._wp_device, dtype=float)
        self._wp_torque_lim = wp.array(torque_lim, device=self._wp_device, dtype=float)
        return

    def pre_step_update(self):
        # Push public root/DOF tensors back into MuJoCo qpos before stepping.
        if (self._num_objs > 0):
            wp.launch(
                kernel=write_root_rot_kernel,
                dim=self._num_envs * self._num_objs,
                inputs=[self._wp_root_rot, self._wp_data.qpos, self._wp_root_qpos_start,
                        self._num_objs],
                device=self._wp_device,
            )

        if (self._num_dofs > 0):
            wp.launch(
                kernel=write_dof_pos_kernel,
                dim=self._num_envs * self._num_dofs,
                inputs=[self._wp_dof_pos, self._wp_data.qpos, self._wp_dof_qpos_start,
                        self._wp_dof_qpos_kind, self._wp_dof_qpos_comp, self._num_dofs],
                device=self._wp_device,
            )
        return

    def post_step_update(self):
        # Refresh the public tensors from MuJoCo-Warp after FK has updated
        # body transforms, velocities, and quaternion-backed DOFs.
        if (self._num_objs > 0):
            wp.launch(
                kernel=update_root_state_kernel,
                dim=self._num_envs * self._num_objs,
                inputs=[self._wp_data.qpos, self._wp_data.qvel, self._wp_data.xquat,
                        self._wp_root_qpos_start, self._wp_root_qvel_start, self._wp_root_body_id,
                        self._wp_root_rot, self._wp_root_vel, self._wp_root_ang_vel,
                        self._num_objs],
                device=self._wp_device,
            )

        if (self._num_dofs > 0):
            wp.launch(
                kernel=update_dof_state_kernel,
                dim=self._num_envs * self._num_dofs,
                inputs=[self._wp_data.qpos, self._wp_data.qvel, self._wp_dof_qpos_start,
                        self._wp_dof_qpos_kind, self._wp_dof_qpos_comp, self._wp_dof_qvel_id,
                        self._wp_dof_pos, self._wp_dof_vel, self._num_dofs],
                device=self._wp_device,
            )

        if (self._num_bodies > 0):
            wp.launch(
                kernel=update_body_state_kernel,
                dim=self._num_envs * self._num_bodies,
                inputs=[self._wp_data.xpos, self._wp_data.xquat, self._wp_data.cvel,
                        self._wp_data.subtree_com, self._wp_body_ids, self._wp_body_root_ids,
                        self._wp_body_pos, self._wp_body_rot, self._wp_body_vel,
                        self._wp_body_ang_vel, self._num_bodies],
                device=self._wp_device,
            )
        return


class Controls:
    def __init__(self, obj_metas, num_envs, device, wp_device, native_pos_control):
        total_dofs = sum(len(meta.qvel_ids) for meta in obj_metas)

        self.target_pos = []
        self.target_vel = []
        self.joint_force = []

        if (total_dofs > 0 and native_pos_control):
            self._wp_target_pos = wp.zeros((num_envs, total_dofs), device=wp_device, dtype=float)
            self._wp_target_vel = None
            self._wp_joint_force = wp.zeros((num_envs, total_dofs), device=wp_device, dtype=float)
            dof_actuator_ids = []
            for meta in obj_metas:
                dof_actuator_ids += meta.actuator_ids
            self._wp_dof_actuator_id = wp.array(dof_actuator_ids, device=wp_device, dtype=int)

            torch_target_pos = wp.to_torch(self._wp_target_pos)
            torch_joint_force = wp.to_torch(self._wp_joint_force)
            torch_target_vel = torch.zeros([num_envs, total_dofs], device=device, dtype=torch.float32)
        elif (total_dofs > 0):
            self._wp_target_pos = wp.zeros((num_envs, total_dofs), device=wp_device, dtype=float)
            self._wp_target_vel = wp.zeros((num_envs, total_dofs), device=wp_device, dtype=float)
            self._wp_joint_force = wp.zeros((num_envs, total_dofs), device=wp_device, dtype=float)
            self._wp_dof_actuator_id = None

            torch_target_pos = wp.to_torch(self._wp_target_pos)
            torch_target_vel = wp.to_torch(self._wp_target_vel)
            torch_joint_force = wp.to_torch(self._wp_joint_force)
        else:
            self._wp_target_pos = None
            self._wp_target_vel = None
            self._wp_joint_force = None
            self._wp_dof_actuator_id = None

            torch_target_pos = torch.zeros([num_envs, 0], device=device, dtype=torch.float32)
            torch_target_vel = torch.zeros([num_envs, 0], device=device, dtype=torch.float32)
            torch_joint_force = torch.zeros([num_envs, 0], device=device, dtype=torch.float32)

        dof_offset = 0
        for meta in obj_metas:
            num_dofs = len(meta.qvel_ids)

            self.target_pos.append(torch_target_pos[:, dof_offset:dof_offset + num_dofs])
            self.joint_force.append(torch_joint_force[:, dof_offset:dof_offset + num_dofs])
            self.target_vel.append(torch_target_vel[:, dof_offset:dof_offset + num_dofs])
            dof_offset += num_dofs
        return


class MujocoEngine(engine.Engine):
    def __init__(self, config, num_envs, device, visualize, record_video=False):
        super().__init__(visualize=visualize)

        self._device = device
        self._wp_device = wp.get_device(device)
        self._num_envs = num_envs
        self._env_spacing = config["env_spacing"]
        self._visualize_enabled = visualize

        sim_freq = config.get("sim_freq", 240)
        control_freq = config.get("control_freq", 30)
        assert(sim_freq >= control_freq and sim_freq % control_freq == 0), \
            "Simulation frequency must be a multiple of the control frequency"

        self._timestep = 1.0 / control_freq
        self._sim_timestep = 1.0 / sim_freq
        self._sim_steps = int(sim_freq / control_freq)
        self._sim_step_count = 0

        self._integrator = config.get("integrator", "implicitfast")
        self._solver = config.get("solver", "newton")
        self._iterations = int(config.get("iterations", 100))
        self._ls_iterations = int(config.get("ls_iterations", 50))
        self._impratio = float(config.get("impratio", 10.0))
        self._cone = config.get("cone", "pyramidal")
        self._jacobian = config.get("jacobian", "auto")
        self._nconmax = config.get("nconmax", 150)
        self._njmax = config.get("njmax", 450)

        if ("control_mode" in config):
            self._control_mode = engine.ControlMode[config["control_mode"]]
        else:
            self._control_mode = engine.ControlMode.none

        self._env_obj_defs = []
        self._env_start_pos = []
        self._env_start_rot = []
        self._env_colors = []

        self._obj_metas = []

        self._camera_pos = np.array([0.0, -5.0, 3.0], dtype=np.float64)
        self._camera_look_at = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self._line_queue = []
        self._keyboard_callbacks = dict()

        self._viewer = None
        self._viewer_data = None
        self._viewer_aux_data = None
        self._viewer_opt = None
        self._viewer_pert = None

        if (visualize):
            record_video = False

        self._record_video = record_video
        self._recording = False
        self._video_recorder = None
        self._state_dirty = False
        self._has_body_forces = False
        self._graph = None
        return

    def get_name(self):
        return "mujoco"

    def create_env(self):
        env_id = len(self._env_obj_defs)
        assert(env_id < self.get_num_envs())

        self._env_obj_defs.append([])
        self._env_start_pos.append([])
        self._env_start_rot.append([])
        self._env_colors.append([])
        return env_id

    def create_obj(self, env_id, obj_type, asset_file, name, is_visual=False, enable_self_collisions=True,
                   fix_root=False, start_pos=None, start_rot=None, color=None, disable_motors=False):
        if (start_pos is None):
            start_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            start_pos = np.asarray(start_pos, dtype=np.float32)

        if (start_rot is None):
            start_rot = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        else:
            start_rot = np.asarray(start_rot, dtype=np.float32)

        obj_def = ObjDef(asset_file=asset_file,
                         name=name,
                         obj_type=obj_type,
                         is_visual=is_visual,
                         enable_self_collisions=enable_self_collisions,
                         fix_root=fix_root,
                         disable_motors=disable_motors)

        obj_id = len(self._env_obj_defs[env_id])
        if (env_id > 0):
            assert(obj_id < len(self._env_obj_defs[0]))
            assert(obj_def == self._env_obj_defs[0][obj_id]), \
                "All MuJoCo envs must create the same objects in the same order."

        self._env_obj_defs[env_id].append(obj_def)
        self._env_start_pos[env_id].append(start_pos)
        self._env_start_rot[env_id].append(start_rot)
        self._env_colors[env_id].append(None if color is None else np.asarray(color, dtype=np.float32))
        return obj_id

    def initialize_sim(self):
        self._validate_envs()
        self._build_model()
        self._build_sim_tensors()
        self._build_controls()
        self._build_dof_force_tensors()
        self._apply_start_xform()
        self._forward()
        self._build_contact_tensors()

        if (self._visualize_enabled):
            self._build_viewer()

        if (self.enabled_record_video()):
            self._video_recorder = self._build_video_recorder()

        self._build_graphs()
        return

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

    def step(self):
        if (self.enabled_record_video() and self._recording):
            self._video_recorder.capture_frame()

        if (self._graph):
            wp.capture_launch(self._graph)
        else:
            self._simulate()

        if (hasattr(self, "_wp_total_contact_forces")):
            self._contact_forces_dirty = True
            self._ground_contact_forces_dirty = True

        if (self._has_body_forces):
            self._torch_xfrc_applied.zero_()
            self._has_body_forces = False

        self._sim_step_count += 1
        return

    def set_camera_pose(self, pos, look_at):
        self._camera_pos = np.asarray(pos, dtype=np.float64)
        self._camera_look_at = np.asarray(look_at, dtype=np.float64)
        self._apply_viewer_camera()
        return

    def get_camera_pos(self):
        return self._camera_pos.copy()

    def get_camera_dir(self):
        cam_dir = self._camera_look_at - self._camera_pos
        norm = np.linalg.norm(cam_dir)
        if (norm > 0.0):
            cam_dir /= norm
        return cam_dir

    def render(self):
        if (self._viewer is None or not self._viewer.is_running()):
            return

        self._sync_derived_state()
        self._viewer.user_scn.ngeom = 0

        self._copy_env_state_to_mjdata(self._viewer_data, 0, self._get_env_offset(0))
        mujoco.mj_forward(self._mj_model, self._viewer_data)
        self._draw_queued_lines(self._viewer.user_scn)
        self._render_extra_envs()

        self._viewer.sync()
        self._line_queue.clear()

        super().render()
        return

    def get_timestep(self):
        return self._timestep

    def get_sim_timestep(self):
        return self._sim_timestep

    def get_sim_time(self):
        return self._sim_step_count * self.get_timestep()

    def get_num_envs(self):
        return self._num_envs

    def get_gravity(self):
        return np.array(self._mj_model.opt.gravity)

    def get_objs_per_env(self):
        return len(self._obj_metas)

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
        self._sync_contact_forces()
        return self._contact_forces[obj_id]

    def get_ground_contact_forces(self, obj_id):
        self._sync_ground_contact_forces()
        return self._ground_contact_forces[obj_id]

    def set_root_pos(self, env_id, obj_id, root_pos):
        root_pos = _as_torch(root_pos, self._device)
        _assign_env_tensor(self._sim_state.root_pos[obj_id], env_id, root_pos)
        self._state_dirty = True
        return

    def set_root_rot(self, env_id, obj_id, root_rot):
        meta = self._obj_metas[obj_id]
        root_rot = _as_torch(root_rot, self._device)
        _assign_env_tensor(self._sim_state.root_rot[obj_id], env_id, root_rot)
        root_rot = _quat_xyzw_to_wxyz(root_rot)

        if (len(meta.root_qpos_ids) > 0):
            target = self._torch_qpos[:, meta.root_qpos_ids[0] + 3:meta.root_qpos_ids[0] + 7]
            _assign_env_tensor(target, env_id, root_rot)

        self._state_dirty = True
        return

    def set_root_vel(self, env_id, obj_id, root_vel):
        meta = self._obj_metas[obj_id]
        root_vel = _as_torch(root_vel, self._device)
        _assign_env_tensor(self._sim_state.root_vel[obj_id], env_id, root_vel)

        if (len(meta.root_qvel_ids) > 0):
            target = self._torch_qvel[:, meta.root_qvel_ids[0]:meta.root_qvel_ids[0] + 3]
            _assign_env_tensor(target, env_id, root_vel)

        self._state_dirty = True
        return

    def set_root_ang_vel(self, env_id, obj_id, root_ang_vel):
        meta = self._obj_metas[obj_id]
        root_ang_vel = _as_torch(root_ang_vel, self._device)
        _assign_env_tensor(self._sim_state.root_ang_vel[obj_id], env_id, root_ang_vel)

        if (len(meta.root_qvel_ids) > 0):
            root_rot = self.get_root_rot(obj_id)
            if (root_ang_vel.dim() == 0):
                if (env_id is None):
                    root_ang_vel = torch.zeros_like(self._torch_qvel[:, meta.root_qvel_ids[0] + 3:meta.root_qvel_ids[0] + 6])
                elif (isinstance(env_id, torch.Tensor)):
                    root_ang_vel = torch.zeros([env_id.shape[0], 3], device=self._device, dtype=torch.float32)
                else:
                    root_ang_vel = torch.zeros([3], device=self._device, dtype=torch.float32)

            if (env_id is None):
                ang_vel_b = torch_util.quat_rotate(torch_util.quat_conjugate(root_rot), root_ang_vel)
                target = self._torch_qvel[:, meta.root_qvel_ids[0] + 3:meta.root_qvel_ids[0] + 6]
                target[:] = ang_vel_b
            else:
                env_rot = root_rot[env_id]
                ang_vel_b = torch_util.quat_rotate(torch_util.quat_conjugate(env_rot), root_ang_vel)
                self._torch_qvel[env_id, meta.root_qvel_ids[0] + 3:meta.root_qvel_ids[0] + 6] = ang_vel_b

        self._state_dirty = True
        return

    def set_dof_pos(self, env_id, obj_id, dof_pos):
        dof_pos = _as_torch(dof_pos, self._device)
        _assign_env_tensor(self._sim_state.dof_pos[obj_id], env_id, dof_pos)
        self._write_dof_pos(self._obj_metas[obj_id], env_id, dof_pos)
        self._state_dirty = True
        return

    def set_dof_vel(self, env_id, obj_id, dof_vel):
        meta = self._obj_metas[obj_id]
        dof_vel = _as_torch(dof_vel, self._device)
        _assign_env_tensor(self._sim_state.dof_vel[obj_id], env_id, dof_vel)
        if (len(meta.qvel_ids) > 0):
            _assign_indexed_cols(self._torch_qvel, meta.qvel_col_ids, env_id, dof_vel)
        self._state_dirty = True
        return

    def set_body_pos(self, env_id, obj_id, body_pos):
        body_pos = _as_torch(body_pos, self._device)
        _assign_env_tensor(self._sim_state.body_pos[obj_id], env_id, body_pos)
        return

    def set_body_rot(self, env_id, obj_id, body_rot):
        body_rot = _as_torch(body_rot, self._device)
        _assign_env_tensor(self._sim_state.body_rot[obj_id], env_id, body_rot)
        return

    def set_body_vel(self, env_id, obj_id, body_vel):
        body_vel = _as_torch(body_vel, self._device)
        _assign_env_tensor(self._sim_state.body_vel[obj_id], env_id, body_vel)
        return

    def set_body_ang_vel(self, env_id, obj_id, body_ang_vel):
        body_ang_vel = _as_torch(body_ang_vel, self._device)
        _assign_env_tensor(self._sim_state.body_ang_vel[obj_id], env_id, body_ang_vel)
        return

    def set_body_forces(self, env_id, obj_id, body_id, forces):
        meta = self._obj_metas[obj_id]
        body_global_id = meta.body_ids[body_id]
        forces = _as_torch(forces, self._device)

        target = self._torch_xfrc_applied[:, body_global_id, 0:3]
        _assign_env_tensor(target, env_id, forces)

        if (env_id is None or not hasattr(env_id, "__len__") or len(env_id) > 0):
            self._has_body_forces = True
        return

    def get_obj_torque_limits(self, env_id, obj_id):
        del env_id
        return self._obj_metas[obj_id].torque_lim.copy()

    def get_obj_dof_limits(self, env_id, obj_id):
        del env_id
        meta = self._obj_metas[obj_id]
        return meta.dof_low.copy(), meta.dof_high.copy()

    def get_obj_pd_gains(self, env_id, obj_id):
        del env_id
        meta = self._obj_metas[obj_id]
        return meta.kp.copy(), meta.kd.copy()

    def find_obj_body_id(self, obj_id, body_name):
        body_names = self.get_obj_body_names(obj_id)
        return body_names.index(body_name)

    def get_obj_type(self, obj_id):
        return self._obj_metas[obj_id].obj_def.obj_type

    def get_obj_num_bodies(self, obj_id):
        return len(self._obj_metas[obj_id].body_ids)

    def get_obj_num_dofs(self, obj_id):
        return len(self._obj_metas[obj_id].qvel_ids)

    def get_obj_body_names(self, obj_id):
        return list(self._obj_metas[obj_id].body_names)

    def calc_obj_mass(self, env_id, obj_id):
        del env_id
        return self._obj_metas[obj_id].mass

    def get_control_mode(self):
        return self._control_mode

    def draw_lines(self, env_id, start_verts, end_verts, cols, line_width):
        self._line_queue.append((env_id, np.asarray(start_verts), np.asarray(end_verts), np.asarray(cols), line_width))
        return

    def register_keyboard_callback(self, key_str, callback_func):
        key_code = str_to_key_code(key_str)
        assert(key_code not in self._keyboard_callbacks)
        self._keyboard_callbacks[key_code] = callback_func
        return

    def enabled_record_video(self):
        return self._record_video

    def get_video_recording(self):
        return self._video_recorder.get_video()

    def start_video_recording(self):
        self._video_recorder.clear()
        self._recording = True
        return

    def stop_video_recording(self):
        self._recording = False
        return

    def get_mj_model(self):
        return self._mj_model

    def get_wp_data(self):
        return self._wp_data

    def create_mj_data(self):
        return mujoco.MjData(self._mj_model)

    def copy_env_state_to_mjdata(self, target_data, env_id, offset=None):
        self._sync_derived_state()
        self._copy_env_state_to_mjdata(target_data, env_id, offset)
        return

    def _validate_envs(self):
        num_envs = self.get_num_envs()
        assert(len(self._env_obj_defs) == num_envs), "Not all MuJoCo envs were created."
        objs_per_env = len(self._env_obj_defs[0])

        for env_id in range(num_envs):
            assert(len(self._env_obj_defs[env_id]) == objs_per_env), \
                "All MuJoCo envs must have the same number of objects."
            for obj_id in range(objs_per_env):
                assert(self._env_obj_defs[env_id][obj_id] == self._env_obj_defs[0][obj_id]), \
                    "All MuJoCo envs must create the same objects in the same order."
        return

    def _build_model(self):
        # Build one MuJoCo model for a single env layout. MuJoCo-Warp owns the
        # replicated per-env data arrays, matching Newton's single model/many worlds flow.
        scene_spec = self._create_scene_spec()
        attached_specs = []

        for obj_id, obj_def in enumerate(self._env_obj_defs[0]):
            color = self._env_colors[0][obj_id]
            obj_spec = self._create_obj_spec(obj_def, color)
            frame = scene_spec.worldbody.add_frame()
            prefix = "obj{:d}_{:s}/".format(obj_id, obj_def.name)
            scene_spec.attach(obj_spec, prefix=prefix, frame=frame)
            attached_specs.append((obj_def, obj_spec, prefix))

        self._mj_model = scene_spec.compile()
        self._apply_model_options()

        self._obj_metas = []
        for obj_def, obj_spec, prefix in attached_specs:
            meta = self._build_obj_meta(obj_def, obj_spec, prefix)
            self._obj_metas.append(meta)

        self._disable_passive_joint_gains()
        self._configure_native_actuators()

        self._mj_data = mujoco.MjData(self._mj_model)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        with wp.ScopedDevice(self._wp_device):
            self._wp_model = mjwarp.put_model(self._mj_model)
            if (hasattr(self._wp_model.opt, "ls_parallel")):
                self._wp_model.opt.ls_parallel = True
            self._wp_data = mjwarp.put_data(
                self._mj_model,
                self._mj_data,
                nworld=self.get_num_envs(),
                nconmax=self._nconmax,
                njmax=self._njmax,
            )
        return

    def _create_scene_spec(self):
        scene_xml = """
<mujoco model="mimickit_mujoco">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="135" elevation="-25" offwidth="1920" offheight="1080"/>
  </visual>
  <worldbody>
    <geom name="ground" type="plane" size="100 100 0.1" condim="3" friction="1 0.05 0.05"/>
  </worldbody>
</mujoco>
"""
        return mujoco.MjSpec.from_string(scene_xml)

    def _create_obj_spec(self, obj_def, color):
        _, file_ext = os.path.splitext(obj_def.asset_file)
        assert(file_ext in [".xml", ".urdf"]), "Unsupported asset format for MuJoCo: {:s}".format(file_ext)

        obj_spec = mujoco.MjSpec.from_file(obj_def.asset_file)

        if (obj_def.fix_root):
            # Match the other engines: a fixed-root object simply has no free joint.
            for joint in list(obj_spec.joints):
                if (int(joint.type) == int(mujoco.mjtJoint.mjJNT_FREE)):
                    obj_spec.delete(joint)

        if (obj_def.disable_motors):
            for actuator in list(obj_spec.actuators):
                obj_spec.delete(actuator)

        if (obj_def.is_visual):
            for geom in obj_spec.geoms:
                geom.contype = 0
                geom.conaffinity = 0

        if (not obj_def.enable_self_collisions):
            self._disable_self_collisions(obj_spec)

        if (color is not None):
            rgba = np.ones(4, dtype=np.float32)
            rgba[:min(3, len(color))] = color[:min(3, len(color))]
            for geom in obj_spec.geoms:
                geom.rgba[:] = rgba

        return obj_spec

    def _disable_self_collisions(self, obj_spec):
        bodies = [b for b in obj_spec.bodies[1:] if b.name]
        for i in range(len(bodies)):
            for j in range(i + 1, len(bodies)):
                obj_spec.add_exclude(bodyname1=bodies[i].name, bodyname2=bodies[j].name)
        return

    def _apply_model_options(self):
        integrator_map = {
            "euler": mujoco.mjtIntegrator.mjINT_EULER,
            "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        }
        solver_map = {
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "pgs": mujoco.mjtSolver.mjSOL_PGS,
        }
        cone_map = {
            "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
            "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
        }
        jacobian_map = {
            "auto": mujoco.mjtJacobian.mjJAC_AUTO,
            "dense": mujoco.mjtJacobian.mjJAC_DENSE,
            "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,
        }

        self._mj_model.opt.timestep = self._sim_timestep
        self._mj_model.opt.integrator = integrator_map[self._integrator]
        self._mj_model.opt.solver = solver_map[self._solver]
        self._mj_model.opt.cone = cone_map[self._cone]
        self._mj_model.opt.jacobian = jacobian_map[self._jacobian]
        self._mj_model.opt.iterations = self._iterations
        self._mj_model.opt.ls_iterations = self._ls_iterations
        self._mj_model.opt.impratio = self._impratio
        return

    def _build_obj_meta(self, obj_def, obj_spec, prefix):
        # Translate MuJoCo ids/ranges into the per-object indexing used by the Engine API.
        body_ids = []
        body_names = []
        for body in obj_spec.bodies[1:]:
            body_name = _strip_prefix(body.name, prefix)
            body_ids.append(int(body.id))
            body_names.append(body_name)

        if (len(body_ids) == 0):
            body_ids = [1]
            body_names = [obj_def.name]

        root_body_id = body_ids[0]
        joint_ids = [int(joint.id) for joint in obj_spec.joints]
        joint_ids = [j for j in joint_ids
                     if int(self._mj_model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_FREE)]

        free_joint_ids = [int(joint.id) for joint in obj_spec.joints
                          if int(self._mj_model.jnt_type[int(joint.id)]) == int(mujoco.mjtJoint.mjJNT_FREE)]
        if (len(free_joint_ids) > 0):
            free_joint_id = free_joint_ids[0]
            root_qpos_start = int(self._mj_model.jnt_qposadr[free_joint_id])
            root_qvel_start = int(self._mj_model.jnt_dofadr[free_joint_id])
            root_qpos_ids = list(range(root_qpos_start, root_qpos_start + 7))
            root_qvel_ids = list(range(root_qvel_start, root_qvel_start + 6))
        else:
            root_qpos_ids = []
            root_qvel_ids = []

        qvel_ids = []
        actuator_ids = []
        hinge_qpos_ids = []
        hinge_dof_ids = []
        ball_qpos_ids = []
        ball_dof_ids = []
        dof_low = []
        dof_high = []
        kp = []
        kd = []
        torque_lim = []

        for joint_id in joint_ids:
            joint_type = int(self._mj_model.jnt_type[joint_id])
            qpos_start = int(self._mj_model.jnt_qposadr[joint_id])
            qvel_start = int(self._mj_model.jnt_dofadr[joint_id])
            qpos_width = _joint_qpos_width(joint_type)
            dof_width = _joint_dof_width(joint_type)

            qvel_ids.extend(range(qvel_start, qvel_start + dof_width))
            joint_actuator_id = self._get_joint_actuator_id(joint_id) if dof_width == 1 else -1
            actuator_ids.extend([joint_actuator_id] * dof_width)

            joint_kp = float(self._mj_model.jnt_stiffness[joint_id])
            joint_kd = np.asarray(self._mj_model.dof_damping[qvel_start:qvel_start + dof_width], dtype=np.float32)
            joint_lim = self._get_joint_torque_limit(joint_id, dof_width)

            if (joint_type == int(mujoco.mjtJoint.mjJNT_BALL)):
                ball_qpos_ids.append(qpos_start)
                ball_dof_ids.append(qvel_start)
                low = np.full(dof_width, -np.pi, dtype=np.float32)
                high = np.full(dof_width, np.pi, dtype=np.float32)
            else:
                hinge_qpos_ids.extend(range(qpos_start, qpos_start + qpos_width))
                hinge_dof_ids.extend(range(qvel_start, qvel_start + dof_width))
                low, high = self._get_joint_pos_limit(joint_id, dof_width)

            dof_low.extend(low)
            dof_high.extend(high)
            kp.extend([joint_kp] * dof_width)
            kd.extend(joint_kd.tolist())
            torque_lim.extend(joint_lim.tolist())

        ball_local_starts = [qvel_ids.index(i) for i in ball_dof_ids]
        qvel_col_ids = torch.tensor(qvel_ids, device=self._device, dtype=torch.long)
        hinge_qpos_col_ids = torch.tensor(hinge_qpos_ids, device=self._device, dtype=torch.long)
        hinge_local_col_ids = torch.tensor([qvel_ids.index(i) for i in hinge_dof_ids],
                                           device=self._device, dtype=torch.long)

        mass = float(np.sum(self._mj_model.body_mass[body_ids]))

        meta = ObjMeta(obj_def=obj_def,
                       body_ids=body_ids,
                       body_names=body_names,
                       qvel_ids=qvel_ids,
                       qvel_col_ids=qvel_col_ids,
                       actuator_ids=actuator_ids,
                       hinge_qpos_ids=hinge_qpos_ids,
                       hinge_dof_ids=hinge_dof_ids,
                       hinge_qpos_col_ids=hinge_qpos_col_ids,
                       hinge_local_col_ids=hinge_local_col_ids,
                       ball_qpos_ids=ball_qpos_ids,
                       ball_dof_ids=ball_dof_ids,
                       ball_local_starts=ball_local_starts,
                       root_body_id=root_body_id,
                       root_qpos_ids=root_qpos_ids,
                       root_qvel_ids=root_qvel_ids,
                       dof_low=np.asarray(dof_low, dtype=np.float32),
                       dof_high=np.asarray(dof_high, dtype=np.float32),
                       kp=np.asarray(kp, dtype=np.float32),
                       kd=np.asarray(kd, dtype=np.float32),
                       torque_lim=np.asarray(torque_lim, dtype=np.float32),
                       mass=mass)
        return meta

    def _get_joint_actuator_id(self, joint_id):
        if (self._mj_model.nu == 0):
            return -1

        joint_actuator = np.logical_and(
            self._mj_model.actuator_trntype == int(mujoco.mjtTrn.mjTRN_JOINT),
            self._mj_model.actuator_trnid[:, 0] == joint_id,
        )
        actuator_ids = np.nonzero(joint_actuator)[0]
        if (len(actuator_ids) != 1):
            return -1
        return int(actuator_ids[0])

    def _get_joint_pos_limit(self, joint_id, dof_width):
        limited = bool(self._mj_model.jnt_limited[joint_id])
        joint_range = np.asarray(self._mj_model.jnt_range[joint_id], dtype=np.float32)
        if (limited and np.isfinite(joint_range).all() and joint_range[0] < joint_range[1]):
            low = np.full(dof_width, joint_range[0], dtype=np.float32)
            high = np.full(dof_width, joint_range[1], dtype=np.float32)
        else:
            raise ValueError("MuJoCo joint {:d} has no finite position limits.".format(joint_id))
        return low, high

    def _get_joint_torque_limit(self, joint_id, dof_width):
        joint_range = np.asarray(self._mj_model.jnt_actfrcrange[joint_id], dtype=np.float32)
        if (np.isfinite(joint_range).all() and joint_range[0] < joint_range[1]
            and np.max(np.abs(joint_range)) > 0.0):
            limit = float(np.max(np.abs(joint_range)))
            return np.full(dof_width, limit, dtype=np.float32)

        actuator_ids = np.where(self._mj_model.actuator_trnid[:, 0] == joint_id)[0]
        limits = []
        for actuator_id in actuator_ids:
            force_range = np.asarray(self._mj_model.actuator_forcerange[actuator_id], dtype=np.float32)
            gear = float(abs(self._mj_model.actuator_gear[actuator_id, 0]))
            if (np.isfinite(force_range).all() and force_range[0] < force_range[1]
                and np.max(np.abs(force_range)) > 0.0 and gear > 0.0):
                limits.append(float(np.max(np.abs(force_range))) * gear)

        if (len(limits) > 0):
            return np.full(dof_width, max(limits), dtype=np.float32)

        raise ValueError("MuJoCo joint {:d} has no finite torque limits.".format(joint_id))

    def _disable_passive_joint_gains(self):
        self._mj_model.jnt_stiffness[:] = 0.0
        self._mj_model.dof_damping[:] = 0.0
        return

    def _configure_native_actuators(self):
        self._use_native_pos_control = self._can_use_native_pos_control()
        if (not self._use_native_pos_control):
            return

        # For the DeepMimic position-control path, let MuJoCo compute the PD
        # actuator forces natively instead of launching an explicit torque kernel.
        for meta in self._obj_metas:
            for local_id, actuator_id in enumerate(meta.actuator_ids):
                self._mj_model.actuator_gaintype[actuator_id] = int(mujoco.mjtGain.mjGAIN_FIXED)
                self._mj_model.actuator_biastype[actuator_id] = int(mujoco.mjtBias.mjBIAS_AFFINE)

                self._mj_model.actuator_gainprm[actuator_id, :] = 0.0
                self._mj_model.actuator_gainprm[actuator_id, 0] = meta.kp[local_id]

                self._mj_model.actuator_biasprm[actuator_id, :] = 0.0
                self._mj_model.actuator_biasprm[actuator_id, 1] = -meta.kp[local_id]
                self._mj_model.actuator_biasprm[actuator_id, 2] = -meta.kd[local_id]

                self._mj_model.actuator_gear[actuator_id, :] = 0.0
                self._mj_model.actuator_gear[actuator_id, 0] = 1.0

                self._mj_model.actuator_ctrlrange[actuator_id, 0] = meta.dof_low[local_id]
                self._mj_model.actuator_ctrlrange[actuator_id, 1] = meta.dof_high[local_id]
                self._mj_model.actuator_ctrllimited[actuator_id] = 1

                limit = meta.torque_lim[local_id]
                self._mj_model.actuator_forcerange[actuator_id, 0] = -limit
                self._mj_model.actuator_forcerange[actuator_id, 1] = limit
                self._mj_model.actuator_forcelimited[actuator_id] = 1
        return

    def _can_use_native_pos_control(self):
        if (self._control_mode != engine.ControlMode.pos):
            return False

        for meta in self._obj_metas:
            if (meta.obj_def.disable_motors and len(meta.qvel_ids) > 0):
                return False
            if (len(meta.ball_dof_ids) > 0):
                return False
            if (len(meta.actuator_ids) != len(meta.qvel_ids)):
                return False
            if (any(actuator_id < 0 for actuator_id in meta.actuator_ids)):
                return False
        return True

    def _build_sim_tensors(self):
        self._sim_state = SimState(self._wp_data, self._obj_metas, self.get_num_envs(), self._device, self._wp_device)
        self._torch_qpos = self._sim_state._torch_qpos
        self._torch_qvel = self._sim_state._torch_qvel
        self._torch_xfrc_applied = self._sim_state._torch_xfrc_applied
        self._torch_ctrl = self._sim_state._torch_ctrl

        self._wp_data.qfrc_applied.zero_()
        self._torch_xfrc_applied.zero_()
        if (self._torch_ctrl.numel() > 0):
            self._torch_ctrl.zero_()
        self._sim_state.post_step_update()
        return

    def _build_controls(self):
        num_envs = self.get_num_envs()
        self._controls = Controls(self._obj_metas, num_envs, self._device, self._wp_device,
                                  self._use_native_pos_control)
        return

    def _build_dof_force_tensors(self):
        self._dof_forces = self._controls.joint_force
        return

    def _apply_start_xform(self):
        num_envs = self.get_num_envs()
        for env_id in range(num_envs):
            for obj_id in range(self.get_objs_per_env()):
                start_pos = torch.tensor(self._env_start_pos[env_id][obj_id], device=self._device, dtype=torch.float32)
                start_rot = torch.tensor(self._env_start_rot[env_id][obj_id], device=self._device, dtype=torch.float32)
                self.set_root_pos(env_id, obj_id, start_pos)
                self.set_root_rot(env_id, obj_id, start_rot)
        return

    def _build_contact_tensors(self):
        num_objs = self.get_objs_per_env()
        max_bodies = max([self.get_obj_num_bodies(i) for i in range(num_objs)] + [1])

        self._wp_contact_ids = wp.array(np.arange(self._wp_data.naconmax), device=self._wp_device, dtype=int)
        self._wp_contact_spatial = wp.zeros(self._wp_data.naconmax, device=self._wp_device, dtype=wp.spatial_vector)
        self._wp_total_contact_forces = wp.zeros((num_objs, self.get_num_envs(), max_bodies, 3),
                                                       device=self._wp_device, dtype=float)
        self._wp_ground_contact_forces = wp.zeros((num_objs, self.get_num_envs(), max_bodies, 3),
                                                        device=self._wp_device, dtype=float)

        body_obj_id = np.full(self._mj_model.nbody, -1, dtype=np.int32)
        body_local_id = np.full(self._mj_model.nbody, -1, dtype=np.int32)
        for obj_id, meta in enumerate(self._obj_metas):
            for local_id, body_id in enumerate(meta.body_ids):
                body_obj_id[body_id] = obj_id
                body_local_id[body_id] = local_id

        self._wp_body_obj_id = wp.array(body_obj_id, device=self._wp_device, dtype=int)
        self._wp_body_local_id = wp.array(body_local_id, device=self._wp_device, dtype=int)

        contact_forces = wp.to_torch(self._wp_total_contact_forces)
        ground_forces = wp.to_torch(self._wp_ground_contact_forces)

        self._contact_forces = []
        self._ground_contact_forces = []
        for obj_id, meta in enumerate(self._obj_metas):
            num_bodies = len(meta.body_ids)
            self._contact_forces.append(contact_forces[obj_id, :, :num_bodies, :])
            self._ground_contact_forces.append(ground_forces[obj_id, :, :num_bodies, :])

        self._contact_forces_dirty = True
        self._ground_contact_forces_dirty = True
        self._update_contact_forces()
        return

    def _apply_cmd(self):
        if (self._control_mode == engine.ControlMode.none or self._use_native_pos_control):
            return

        if (self._sim_state._num_dofs > 0):
            if (self._control_mode == engine.ControlMode.vel):
                cmd = self._controls._wp_target_vel
            elif (self._control_mode == engine.ControlMode.torque):
                cmd = self._controls._wp_joint_force
            else:
                cmd = self._controls._wp_target_pos

            wp.launch(
                kernel=apply_control_kernel,
                dim=self.get_num_envs() * self._sim_state._num_dofs,
                inputs=[
                    self._wp_data.qpos,
                    self._wp_data.qvel,
                    cmd,
                    self._sim_state._wp_dof_qpos_start,
                    self._sim_state._wp_dof_qpos_kind,
                    self._sim_state._wp_dof_qpos_comp,
                    self._sim_state._wp_control_qvel_id,
                    self._sim_state._wp_kp,
                    self._sim_state._wp_kd,
                    self._sim_state._wp_torque_lim,
                    self._wp_data.qfrc_applied,
                    self._controls._wp_joint_force,
                    self._control_mode.value,
                    self._sim_state._num_dofs,
                ],
                device=self._wp_device,
            )
        return

    def _build_graphs(self):
        self._graph = None
        self._contact_graph = None
        self._ground_contact_graph = None

        if (not self._wp_device.is_cuda):
            return

        try:
            with wp.ScopedDevice(self._wp_device):
                with wp.ScopedCapture() as capture:
                    self._simulate()
            self._graph = capture.graph
        except Exception as e:
            Logger.print("MuJoCo Warp graph capture disabled: {}".format(e))
            self._graph = None

        try:
            with wp.ScopedDevice(self._wp_device):
                with wp.ScopedCapture() as capture:
                    self._update_contact_forces()
                self._contact_graph = capture.graph

                with wp.ScopedCapture() as capture:
                    self._update_ground_contact_forces()
                self._ground_contact_graph = capture.graph
            self._contact_forces_dirty = True
            self._ground_contact_forces_dirty = True
        except Exception as e:
            Logger.print("MuJoCo Warp contact graph capture disabled: {}".format(e))
            self._contact_graph = None
            self._ground_contact_graph = None
        return

    def _simulate(self):
        # Keep the same high-level cadence as Newton: sync API tensors, apply
        # controls each substep, step the solver, then refresh derived tensors.
        self._sim_state.pre_step_update()
        if (self._use_native_pos_control):
            self._write_native_ctrl()

        for _ in range(self._sim_steps):
            self._apply_cmd()
            with wp.ScopedDevice(self._wp_device):
                mjwarp.step(self._wp_model, self._wp_data)

        self._sim_state.post_step_update()
        if (self._use_native_pos_control):
            self._update_native_dof_forces()
        self._contact_forces_dirty = True
        self._ground_contact_forces_dirty = True
        self._state_dirty = False
        return

    def _write_native_ctrl(self):
        if (self._sim_state._num_dofs > 0):
            wp.launch(
                kernel=write_ctrl_kernel,
                dim=self.get_num_envs() * self._sim_state._num_dofs,
                inputs=[
                    self._controls._wp_target_pos,
                    self._controls._wp_dof_actuator_id,
                    self._wp_data.ctrl,
                    self._sim_state._num_dofs,
                ],
                device=self._wp_device,
            )
        return

    def _update_native_dof_forces(self):
        if (self._sim_state._num_dofs > 0):
            wp.launch(
                kernel=update_actuator_force_kernel,
                dim=self.get_num_envs() * self._sim_state._num_dofs,
                inputs=[
                    self._wp_data.actuator_force,
                    self._controls._wp_dof_actuator_id,
                    self._controls._wp_joint_force,
                    self._sim_state._num_dofs,
                ],
                device=self._wp_device,
            )
        return

    def _write_dof_pos(self, meta, env_id, dof_pos):
        if (len(meta.qvel_ids) == 0):
            return

        if (len(meta.hinge_dof_ids) > 0):
            values = dof_pos[..., meta.hinge_local_col_ids]
            _assign_indexed_cols(self._torch_qpos, meta.hinge_qpos_col_ids, env_id, values)

        for qpos_start, local_start in zip(meta.ball_qpos_ids, meta.ball_local_starts):
            quat = torch_util.exp_map_to_quat(dof_pos[..., local_start:local_start + 3])
            quat = _quat_xyzw_to_wxyz(quat)
            _assign_env_tensor(self._torch_qpos[:, qpos_start:qpos_start + 4], env_id, quat)
        return

    def _forward(self):
        self._sim_state.pre_step_update()
        with wp.ScopedDevice(self._wp_device):
            mjwarp.forward(self._wp_model, self._wp_data)
        self._sim_state.post_step_update()
        self._state_dirty = False
        return

    def _sync_derived_state(self):
        if (self._state_dirty):
            self._forward()
        return

    def _sync_contact_forces(self):
        if (getattr(self, "_contact_forces_dirty", False)):
            if (self._contact_graph is not None):
                wp.capture_launch(self._contact_graph)
                self._contact_forces_dirty = False
                self._ground_contact_forces_dirty = False
            else:
                self._update_contact_forces()
        return

    def _sync_ground_contact_forces(self):
        if (getattr(self, "_ground_contact_forces_dirty", False)):
            if (self._ground_contact_graph is not None):
                wp.capture_launch(self._ground_contact_graph)
                self._ground_contact_forces_dirty = False
            else:
                self._update_ground_contact_forces()
        return

    def _update_contact_forces(self):
        if (not hasattr(self, "_wp_total_contact_forces")):
            return

        # MuJoCo-Warp reports contacts as one global contact stream. Reduce it
        # lazily into MimicKit's per-object body-force tensors only when queried.
        self._wp_total_contact_forces.zero_()
        self._wp_ground_contact_forces.zero_()

        if (self._wp_data.naconmax == 0):
            self._contact_forces_dirty = False
            self._ground_contact_forces_dirty = False
            return

        mjwarp_support.contact_force(
            self._wp_model,
            self._wp_data,
            self._wp_contact_ids,
            True,
            self._wp_contact_spatial,
        )

        wp.launch(
            kernel=accumulate_contact_kernel,
            dim=self._wp_data.naconmax,
            inputs=[
                self._wp_data.contact.worldid,
                self._wp_data.contact.geom,
                self._wp_data.contact.type,
                self._wp_contact_spatial,
                self._wp_data.nacon,
                self._wp_model.geom_bodyid,
                self._wp_body_obj_id,
                self._wp_body_local_id,
            ],
            outputs=[
                self._wp_total_contact_forces,
                self._wp_ground_contact_forces,
            ],
            device=self._wp_device,
        )
        self._contact_forces_dirty = False
        self._ground_contact_forces_dirty = False
        return

    def _update_ground_contact_forces(self):
        if (not hasattr(self, "_wp_ground_contact_forces")):
            return

        self._wp_ground_contact_forces.zero_()

        if (self._wp_data.naconmax == 0):
            self._ground_contact_forces_dirty = False
            return

        mjwarp_support.contact_force(
            self._wp_model,
            self._wp_data,
            self._wp_contact_ids,
            True,
            self._wp_contact_spatial,
        )

        wp.launch(
            kernel=accumulate_ground_contact_kernel,
            dim=self._wp_data.naconmax,
            inputs=[
                self._wp_data.contact.worldid,
                self._wp_data.contact.geom,
                self._wp_data.contact.type,
                self._wp_contact_spatial,
                self._wp_data.nacon,
                self._wp_model.geom_bodyid,
                self._wp_body_obj_id,
                self._wp_body_local_id,
            ],
            outputs=[
                self._wp_ground_contact_forces,
            ],
            device=self._wp_device,
        )
        self._ground_contact_forces_dirty = False
        return

    def _build_viewer(self):
        import mujoco.viewer

        self._viewer_data = mujoco.MjData(self._mj_model)
        self._viewer_aux_data = mujoco.MjData(self._mj_model)
        self._viewer_opt = mujoco.MjvOption()
        self._viewer_pert = mujoco.MjvPerturb()

        self._copy_env_state_to_mjdata(self._viewer_data, 0, self._get_env_offset(0))
        mujoco.mj_forward(self._mj_model, self._viewer_data)

        self._viewer = mujoco.viewer.launch_passive(
            self._mj_model,
            self._viewer_data,
            key_callback=self._on_keyboard_event,
            show_left_ui=False,
            show_right_ui=False,
        )
        self._apply_viewer_camera()
        return

    def _render_extra_envs(self):
        if (self.get_num_envs() <= 1):
            return

        catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC.value
        for env_id in range(1, self.get_num_envs()):
            self._copy_env_state_to_mjdata(self._viewer_aux_data, env_id, self._get_env_offset(env_id))
            mujoco.mj_forward(self._mj_model, self._viewer_aux_data)
            mujoco.mjv_addGeoms(
                self._mj_model,
                self._viewer_aux_data,
                self._viewer_opt,
                self._viewer_pert,
                catmask,
                self._viewer.user_scn,
            )
        return

    def _copy_env_state_to_mjdata(self, target_data, env_id, offset=None):
        if (offset is None):
            offset = np.zeros(3, dtype=np.float64)
        else:
            offset = np.asarray(offset, dtype=np.float64)

        if (self._mj_model.nq > 0):
            target_data.qpos[:] = self._torch_qpos[env_id].detach().cpu().numpy()
            target_data.qvel[:] = self._torch_qvel[env_id].detach().cpu().numpy()

            for meta in self._obj_metas:
                if (len(meta.root_qpos_ids) > 0):
                    root_start = meta.root_qpos_ids[0]
                    target_data.qpos[root_start:root_start + 3] += offset

        if (self._mj_model.nu > 0):
            target_data.ctrl[:] = self._torch_ctrl[env_id].detach().cpu().numpy()

        target_data.xfrc_applied[:] = self._torch_xfrc_applied[env_id].detach().cpu().numpy()
        return

    def _apply_viewer_camera(self):
        if (self._viewer is None):
            return

        cam_vec = self._camera_pos - self._camera_look_at
        distance = max(np.linalg.norm(cam_vec), 1e-5)
        self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE.value
        self._viewer.cam.fixedcamid = -1
        self._viewer.cam.trackbodyid = -1
        self._viewer.cam.lookat[:] = self._camera_look_at + self._get_env_offset(0)
        self._viewer.cam.distance = distance
        self._viewer.cam.azimuth = math.degrees(math.atan2(cam_vec[1], cam_vec[0]))
        self._viewer.cam.elevation = math.degrees(math.asin(np.clip(cam_vec[2] / distance, -1.0, 1.0)))
        return

    def _draw_queued_lines(self, scene):
        for env_id, start_verts, end_verts, cols, line_width in self._line_queue:
            offset = self._get_env_offset(env_id)
            for i in range(start_verts.shape[0]):
                if (scene.ngeom >= scene.maxgeom):
                    return

                rgba = np.ones(4, dtype=np.float32)
                rgba[:min(4, cols.shape[-1])] = cols[i, :min(4, cols.shape[-1])]
                start = np.asarray(start_verts[i], dtype=np.float64) + offset
                end = np.asarray(end_verts[i], dtype=np.float64) + offset

                scene.ngeom += 1
                geom = scene.geoms[scene.ngeom - 1]
                geom.category = mujoco.mjtCatBit.mjCAT_DECOR
                mujoco.mjv_initGeom(
                    geom=geom,
                    type=mujoco.mjtGeom.mjGEOM_LINE.value,
                    size=np.zeros(3, dtype=np.float64),
                    pos=np.zeros(3, dtype=np.float64),
                    mat=np.eye(3, dtype=np.float64).reshape(-1),
                    rgba=rgba,
                )
                mujoco.mjv_connector(
                    geom=geom,
                    type=mujoco.mjtGeom.mjGEOM_LINE.value,
                    width=float(line_width),
                    from_=start,
                    to=end,
                )
        return

    def _get_env_offset(self, env_id):
        if (self.get_num_envs() <= 1):
            return np.zeros(3, dtype=np.float64)

        num_per_row = int(np.ceil(np.sqrt(self.get_num_envs())))
        row = env_id // num_per_row
        col = env_id % num_per_row
        offset = np.array([col * self._env_spacing, row * self._env_spacing, 0.0], dtype=np.float64)
        return offset

    def _on_keyboard_event(self, key):
        if (key in self._keyboard_callbacks):
            self._keyboard_callbacks[key]()
        return

    def _build_video_recorder(self):
        import engines.mujoco_recorder as mujoco_recorder

        Logger.print("Video recording enabled")
        return mujoco_recorder.MujocoVideoRecorder(self)


def _strip_prefix(name, prefix):
    if (name.startswith(prefix)):
        return name[len(prefix):]
    return os.path.basename(name)


def _joint_dof_width(joint_type):
    if (joint_type == int(mujoco.mjtJoint.mjJNT_FREE)):
        return 6
    elif (joint_type == int(mujoco.mjtJoint.mjJNT_BALL)):
        return 3
    elif (joint_type == int(mujoco.mjtJoint.mjJNT_HINGE)):
        return 1
    elif (joint_type == int(mujoco.mjtJoint.mjJNT_SLIDE)):
        return 1
    assert(False), "Unsupported MuJoCo joint type: {}".format(joint_type)


def _joint_qpos_width(joint_type):
    if (joint_type == int(mujoco.mjtJoint.mjJNT_FREE)):
        return 7
    elif (joint_type == int(mujoco.mjtJoint.mjJNT_BALL)):
        return 4
    elif (joint_type == int(mujoco.mjtJoint.mjJNT_HINGE)):
        return 1
    elif (joint_type == int(mujoco.mjtJoint.mjJNT_SLIDE)):
        return 1
    assert(False), "Unsupported MuJoCo joint type: {}".format(joint_type)


def _quat_xyzw_to_wxyz(q):
    return torch.cat([q[..., 3:4], q[..., 0:3]], dim=-1)


def _as_torch(x, device):
    if (isinstance(x, torch.Tensor)):
        return x.to(device=device, dtype=torch.float32)
    return torch.tensor(x, device=device, dtype=torch.float32)


def _assign_env_tensor(target, env_id, value):
    if (env_id is None):
        target[:] = value
    else:
        target[env_id] = value
    return


def _assign_indexed_cols(base, col_ids, env_id, value):
    if (isinstance(col_ids, torch.Tensor)):
        if (col_ids.device != base.device or col_ids.dtype != torch.long):
            col_ids = col_ids.to(device=base.device, dtype=torch.long)
    else:
        col_ids = torch.tensor(col_ids, device=base.device, dtype=torch.long)

    value = value.to(device=base.device, dtype=base.dtype)

    if (env_id is None):
        base[:, col_ids] = value
    elif (isinstance(env_id, torch.Tensor)):
        env_ids = env_id.to(device=base.device, dtype=torch.long)
        base[env_ids.unsqueeze(-1), col_ids.unsqueeze(0)] = value
    else:
        base[env_id, col_ids] = value
    return


def str_to_key_code(key_str):
    key_name = key_str.upper()
    aliases = {
        "RETURN": "ENTER",
        "ESC": "ESCAPE",
        "DEL": "DELETE",
    }
    key_name = aliases.get(key_name, key_name)

    named_keys = {
        "SPACE": 32,
        "ESCAPE": 256,
        "ENTER": 257,
        "TAB": 258,
        "BACKSPACE": 259,
        "INSERT": 260,
        "DELETE": 261,
        "RIGHT": 262,
        "LEFT": 263,
        "DOWN": 264,
        "UP": 265,
    }
    if (key_name in named_keys):
        return named_keys[key_name]
    if (len(key_name) == 1):
        return ord(key_name)
    if (key_name.startswith("F") and key_name[1:].isdigit()):
        return 289 + int(key_name[1:])

    assert(False), "Unsupported MuJoCo key: {:s}".format(key_str)


@wp.func
def quat_wxyz_to_exp_map(q: wp.quat):
    min_theta = 1e-5

    w = q[0]
    axis = wp.vec3f(q[1], q[2], q[3])
    if (w < 0.0):
        w = -w
        axis = -axis

    sin_angle = wp.length(axis)
    angle = 2.0 * wp.atan2(sin_angle, w)

    if (sin_angle > min_theta):
        axis = axis / sin_angle
    else:
        axis = wp.vec3f(0.0, 0.0, 1.0)
        angle = 0.0

    return axis * angle

@wp.func
def exp_map_to_quat_wxyz(exp_map: wp.vec3f):
    min_theta = 1e-5

    angle = wp.length(exp_map)
    axis = wp.vec3f(0.0, 0.0, 1.0)
    if (wp.abs(angle) > min_theta):
        axis = exp_map / angle
        angle = wp.atan2(wp.sin(angle), wp.cos(angle))
    else:
        angle = 0.0

    s = wp.sin(0.5 * angle)
    c = wp.cos(0.5 * angle)
    return wp.quat(c, axis[0] * s, axis[1] * s, axis[2] * s)

@wp.func
def rot_vec_quat_wxyz(vec: wp.vec3f, quat: wp.quat):
    s = quat[0]
    u = wp.vec3f(quat[1], quat[2], quat[3])
    r = 2.0 * (wp.dot(u, vec) * u) + (s * s - wp.dot(u, u)) * vec
    r = r + 2.0 * s * wp.cross(u, vec)
    return r

@wp.kernel
def update_dof_state_kernel(qpos: wp.array2d(dtype=float),
                            qvel: wp.array2d(dtype=float),
                            dof_qpos_start: wp.array(dtype=int),
                            dof_qpos_kind: wp.array(dtype=int),
                            dof_qpos_comp: wp.array(dtype=int),
                            dof_qvel_id: wp.array(dtype=int),
                            dof_pos: wp.array2d(dtype=float),
                            dof_vel: wp.array2d(dtype=float),
                            num_dofs: int):
    tid = wp.tid()
    world_id = tid // num_dofs
    dof_id = tid - world_id * num_dofs

    qvel_id = dof_qvel_id[dof_id]
    if (qvel_id >= 0):
        dof_vel[world_id, dof_id] = qvel[world_id, qvel_id]

    qpos_start = dof_qpos_start[dof_id]
    qpos_kind = dof_qpos_kind[dof_id]
    if (qpos_start < 0):
        return

    if (qpos_kind == 0):
        dof_pos[world_id, dof_id] = qpos[world_id, qpos_start]
    else:
        q = wp.quat(qpos[world_id, qpos_start],
                    qpos[world_id, qpos_start + 1],
                    qpos[world_id, qpos_start + 2],
                    qpos[world_id, qpos_start + 3])
        exp_map = quat_wxyz_to_exp_map(q)
        comp = dof_qpos_comp[dof_id]
        dof_pos[world_id, dof_id] = exp_map[comp]
    return

@wp.kernel
def write_dof_pos_kernel(dof_pos: wp.array2d(dtype=float),
                         qpos: wp.array2d(dtype=float),
                         dof_qpos_start: wp.array(dtype=int),
                         dof_qpos_kind: wp.array(dtype=int),
                         dof_qpos_comp: wp.array(dtype=int),
                         num_dofs: int):
    tid = wp.tid()
    world_id = tid // num_dofs
    dof_id = tid - world_id * num_dofs

    qpos_start = dof_qpos_start[dof_id]
    if (qpos_start < 0):
        return

    qpos_kind = dof_qpos_kind[dof_id]
    if (qpos_kind == 0):
        qpos[world_id, qpos_start] = dof_pos[world_id, dof_id]
    elif (dof_qpos_comp[dof_id] == 0):
        exp_map = wp.vec3f(dof_pos[world_id, dof_id],
                           dof_pos[world_id, dof_id + 1],
                           dof_pos[world_id, dof_id + 2])
        q = exp_map_to_quat_wxyz(exp_map)
        qpos[world_id, qpos_start] = q[0]
        qpos[world_id, qpos_start + 1] = q[1]
        qpos[world_id, qpos_start + 2] = q[2]
        qpos[world_id, qpos_start + 3] = q[3]
    return

@wp.kernel
def write_root_rot_kernel(root_rot: wp.array3d(dtype=float),
                          qpos: wp.array2d(dtype=float),
                          root_qpos_start: wp.array(dtype=int),
                          num_objs: int):
    tid = wp.tid()
    world_id = tid // num_objs
    obj_id = tid - world_id * num_objs

    qpos_start = root_qpos_start[obj_id]
    if (qpos_start >= 0):
        qpos[world_id, qpos_start + 3] = root_rot[world_id, obj_id, 3]
        qpos[world_id, qpos_start + 4] = root_rot[world_id, obj_id, 0]
        qpos[world_id, qpos_start + 5] = root_rot[world_id, obj_id, 1]
        qpos[world_id, qpos_start + 6] = root_rot[world_id, obj_id, 2]
    return

@wp.kernel
def update_root_state_kernel(qpos: wp.array2d(dtype=float),
                             qvel: wp.array2d(dtype=float),
                             xquat: wp.array2d(dtype=wp.quat),
                             root_qpos_start: wp.array(dtype=int),
                             root_qvel_start: wp.array(dtype=int),
                             root_body_id: wp.array(dtype=int),
                             root_rot: wp.array3d(dtype=float),
                             root_vel: wp.array3d(dtype=float),
                             root_ang_vel: wp.array3d(dtype=float),
                             num_objs: int):
    tid = wp.tid()
    world_id = tid // num_objs
    obj_id = tid - world_id * num_objs

    qpos_start = root_qpos_start[obj_id]
    if (qpos_start >= 0):
        qvel_start = root_qvel_start[obj_id]
        q = wp.quat(qpos[world_id, qpos_start + 3],
                    qpos[world_id, qpos_start + 4],
                    qpos[world_id, qpos_start + 5],
                    qpos[world_id, qpos_start + 6])

        root_rot[world_id, obj_id, 0] = q[1]
        root_rot[world_id, obj_id, 1] = q[2]
        root_rot[world_id, obj_id, 2] = q[3]
        root_rot[world_id, obj_id, 3] = q[0]

        root_vel[world_id, obj_id, 0] = qvel[world_id, qvel_start]
        root_vel[world_id, obj_id, 1] = qvel[world_id, qvel_start + 1]
        root_vel[world_id, obj_id, 2] = qvel[world_id, qvel_start + 2]

        ang_vel_b = wp.vec3f(qvel[world_id, qvel_start + 3],
                             qvel[world_id, qvel_start + 4],
                             qvel[world_id, qvel_start + 5])
        ang_vel_w = rot_vec_quat_wxyz(ang_vel_b, q)
        root_ang_vel[world_id, obj_id, 0] = ang_vel_w[0]
        root_ang_vel[world_id, obj_id, 1] = ang_vel_w[1]
        root_ang_vel[world_id, obj_id, 2] = ang_vel_w[2]
    else:
        body_id = root_body_id[obj_id]
        q = xquat[world_id, body_id]
        root_rot[world_id, obj_id, 0] = q[1]
        root_rot[world_id, obj_id, 1] = q[2]
        root_rot[world_id, obj_id, 2] = q[3]
        root_rot[world_id, obj_id, 3] = q[0]

        root_vel[world_id, obj_id, 0] = 0.0
        root_vel[world_id, obj_id, 1] = 0.0
        root_vel[world_id, obj_id, 2] = 0.0

        root_ang_vel[world_id, obj_id, 0] = 0.0
        root_ang_vel[world_id, obj_id, 1] = 0.0
        root_ang_vel[world_id, obj_id, 2] = 0.0
    return

@wp.kernel
def update_body_state_kernel(xpos: wp.array2d(dtype=wp.vec3),
                             xquat: wp.array2d(dtype=wp.quat),
                             cvel: wp.array2d(dtype=wp.spatial_vector),
                             subtree_com: wp.array2d(dtype=wp.vec3),
                             body_ids: wp.array(dtype=int),
                             body_root_ids: wp.array(dtype=int),
                             body_pos: wp.array3d(dtype=float),
                             body_rot: wp.array3d(dtype=float),
                             body_vel: wp.array3d(dtype=float),
                             body_ang_vel: wp.array3d(dtype=float),
                             num_bodies: int):
    tid = wp.tid()
    world_id = tid // num_bodies
    local_body_id = tid - world_id * num_bodies

    body_id = body_ids[local_body_id]
    root_body_id = body_root_ids[local_body_id]

    pos = xpos[world_id, body_id]
    cvel_body = cvel[world_id, body_id]
    ang_vel = wp.vec3f(cvel_body[0], cvel_body[1], cvel_body[2])
    lin_vel_c = wp.vec3f(cvel_body[3], cvel_body[4], cvel_body[5])
    com = subtree_com[world_id, root_body_id]
    lin_vel = lin_vel_c - wp.cross(ang_vel, com - pos)
    q = xquat[world_id, body_id]

    body_pos[world_id, local_body_id, 0] = pos[0]
    body_pos[world_id, local_body_id, 1] = pos[1]
    body_pos[world_id, local_body_id, 2] = pos[2]

    body_rot[world_id, local_body_id, 0] = q[1]
    body_rot[world_id, local_body_id, 1] = q[2]
    body_rot[world_id, local_body_id, 2] = q[3]
    body_rot[world_id, local_body_id, 3] = q[0]

    body_vel[world_id, local_body_id, 0] = lin_vel[0]
    body_vel[world_id, local_body_id, 1] = lin_vel[1]
    body_vel[world_id, local_body_id, 2] = lin_vel[2]

    body_ang_vel[world_id, local_body_id, 0] = ang_vel[0]
    body_ang_vel[world_id, local_body_id, 1] = ang_vel[1]
    body_ang_vel[world_id, local_body_id, 2] = ang_vel[2]
    return

@wp.kernel
def apply_control_kernel(qpos: wp.array2d(dtype=float),
                         qvel: wp.array2d(dtype=float),
                         cmd: wp.array2d(dtype=float),
                         dof_qpos_start: wp.array(dtype=int),
                         dof_qpos_kind: wp.array(dtype=int),
                         dof_qpos_comp: wp.array(dtype=int),
                         dof_qvel_id: wp.array(dtype=int),
                         kp: wp.array(dtype=float),
                         kd: wp.array(dtype=float),
                         torque_lim: wp.array(dtype=float),
                         qfrc_applied: wp.array2d(dtype=float),
                         dof_force: wp.array2d(dtype=float),
                         control_mode: int,
                         num_dofs: int):
    tid = wp.tid()
    world_id = tid // num_dofs
    dof_id = tid - world_id * num_dofs

    qvel_id = dof_qvel_id[dof_id]
    if (qvel_id < 0):
        return

    cmd_val = cmd[world_id, dof_id]
    qvel_val = qvel[world_id, qvel_id]
    torque = 0.0

    if (control_mode == 1 or control_mode == 4):
        qpos_start = dof_qpos_start[dof_id]
        qpos_kind = dof_qpos_kind[dof_id]
        dof_val = 0.0
        if (qpos_kind == 0):
            dof_val = qpos[world_id, qpos_start]
        else:
            q = wp.quat(qpos[world_id, qpos_start],
                        qpos[world_id, qpos_start + 1],
                        qpos[world_id, qpos_start + 2],
                        qpos[world_id, qpos_start + 3])
            exp_map = quat_wxyz_to_exp_map(q)
            dof_val = exp_map[dof_qpos_comp[dof_id]]
        torque = kp[dof_id] * (cmd_val - dof_val) - kd[dof_id] * qvel_val
    elif (control_mode == 2):
        torque = kd[dof_id] * (cmd_val - qvel_val)
    elif (control_mode == 3):
        torque = cmd_val

    limit = torque_lim[dof_id]
    torque = wp.clamp(torque, -limit, limit)
    qfrc_applied[world_id, qvel_id] = torque
    dof_force[world_id, dof_id] = torque
    return

@wp.kernel
def write_ctrl_kernel(cmd: wp.array2d(dtype=float),
                      dof_actuator_id: wp.array(dtype=int),
                      ctrl: wp.array2d(dtype=float),
                      num_dofs: int):
    tid = wp.tid()
    world_id = tid // num_dofs
    dof_id = tid - world_id * num_dofs

    actuator_id = dof_actuator_id[dof_id]
    if (actuator_id >= 0):
        ctrl[world_id, actuator_id] = cmd[world_id, dof_id]
    return

@wp.kernel
def update_actuator_force_kernel(actuator_force: wp.array2d(dtype=float),
                                 dof_actuator_id: wp.array(dtype=int),
                                 dof_force: wp.array2d(dtype=float),
                                 num_dofs: int):
    tid = wp.tid()
    world_id = tid // num_dofs
    dof_id = tid - world_id * num_dofs

    actuator_id = dof_actuator_id[dof_id]
    if (actuator_id >= 0):
        dof_force[world_id, dof_id] = actuator_force[world_id, actuator_id]
    return

@wp.kernel
def accumulate_contact_kernel(contact_worldid: wp.array(dtype=int),
                              contact_geom: wp.array(dtype=wp.vec2i),
                              contact_type: wp.array(dtype=int),
                              contact_force: wp.array(dtype=wp.spatial_vector),
                              nacon: wp.array(dtype=int),
                              geom_bodyid: wp.array(dtype=int),
                              body_obj_id: wp.array(dtype=int),
                              body_local_id: wp.array(dtype=int),
                              total_forces: wp.array4d(dtype=float),
                              ground_forces: wp.array4d(dtype=float)):
    contact_id = wp.tid()

    if (contact_id >= nacon[0]):
        return

    # ContactType.CONSTRAINT == 1 in mujoco_warp.
    if ((contact_type[contact_id] & 1) == 0):
        return

    geom_pair = contact_geom[contact_id]
    geom0 = geom_pair[0]
    geom1 = geom_pair[1]
    if (geom0 < 0 or geom1 < 0):
        return

    world_id = contact_worldid[contact_id]
    body0 = geom_bodyid[geom0]
    body1 = geom_bodyid[geom1]
    obj0 = body_obj_id[body0]
    obj1 = body_obj_id[body1]
    local0 = body_local_id[body0]
    local1 = body_local_id[body1]

    frc = wp.spatial_top(contact_force[contact_id])

    if (obj0 >= 0 and local0 >= 0):
        wp.atomic_add(total_forces, obj0, world_id, local0, 0, frc[0])
        wp.atomic_add(total_forces, obj0, world_id, local0, 1, frc[1])
        wp.atomic_add(total_forces, obj0, world_id, local0, 2, frc[2])

    if (obj1 >= 0 and local1 >= 0):
        wp.atomic_add(total_forces, obj1, world_id, local1, 0, -frc[0])
        wp.atomic_add(total_forces, obj1, world_id, local1, 1, -frc[1])
        wp.atomic_add(total_forces, obj1, world_id, local1, 2, -frc[2])

    if (body0 == 0 and obj1 >= 0 and local1 >= 0):
        wp.atomic_add(ground_forces, obj1, world_id, local1, 0, -frc[0])
        wp.atomic_add(ground_forces, obj1, world_id, local1, 1, -frc[1])
        wp.atomic_add(ground_forces, obj1, world_id, local1, 2, -frc[2])
    elif (body1 == 0 and obj0 >= 0 and local0 >= 0):
        wp.atomic_add(ground_forces, obj0, world_id, local0, 0, frc[0])
        wp.atomic_add(ground_forces, obj0, world_id, local0, 1, frc[1])
        wp.atomic_add(ground_forces, obj0, world_id, local0, 2, frc[2])

    return

@wp.kernel
def accumulate_ground_contact_kernel(contact_worldid: wp.array(dtype=int),
                                     contact_geom: wp.array(dtype=wp.vec2i),
                                     contact_type: wp.array(dtype=int),
                                     contact_force: wp.array(dtype=wp.spatial_vector),
                                     nacon: wp.array(dtype=int),
                                     geom_bodyid: wp.array(dtype=int),
                                     body_obj_id: wp.array(dtype=int),
                                     body_local_id: wp.array(dtype=int),
                                     ground_forces: wp.array4d(dtype=float)):
    contact_id = wp.tid()

    if (contact_id >= nacon[0]):
        return

    if ((contact_type[contact_id] & 1) == 0):
        return

    geom_pair = contact_geom[contact_id]
    geom0 = geom_pair[0]
    geom1 = geom_pair[1]
    if (geom0 < 0 or geom1 < 0):
        return

    world_id = contact_worldid[contact_id]
    body0 = geom_bodyid[geom0]
    body1 = geom_bodyid[geom1]
    obj0 = body_obj_id[body0]
    obj1 = body_obj_id[body1]
    local0 = body_local_id[body0]
    local1 = body_local_id[body1]

    frc = wp.spatial_top(contact_force[contact_id])

    if (body0 == 0 and obj1 >= 0 and local1 >= 0):
        wp.atomic_add(ground_forces, obj1, world_id, local1, 0, -frc[0])
        wp.atomic_add(ground_forces, obj1, world_id, local1, 1, -frc[1])
        wp.atomic_add(ground_forces, obj1, world_id, local1, 2, -frc[2])
    elif (body1 == 0 and obj0 >= 0 and local0 >= 0):
        wp.atomic_add(ground_forces, obj0, world_id, local0, 0, frc[0])
        wp.atomic_add(ground_forces, obj0, world_id, local0, 1, frc[1])
        wp.atomic_add(ground_forces, obj0, world_id, local0, 2, frc[2])

    return

