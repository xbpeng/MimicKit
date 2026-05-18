from __future__ import annotations

from dataclasses import dataclass
import math
import os

import numpy as np
import torch

import engines.engine as engine
from util.logger import Logger
import util.torch_util as torch_util


_mujoco = None
_mjwarp = None
_mjwarp_support = None
_wp = None
_accumulate_contact_kernel = None


def _load_mujoco_modules():
    global _mujoco, _mjwarp, _mjwarp_support, _wp

    if (_mujoco is not None):
        return _mujoco, _mjwarp, _mjwarp_support, _wp

    try:
        import mujoco
        import mujoco_warp as mjwarp
        import warp as wp
        from mujoco_warp._src import support as mjwarp_support
    except ImportError as e:
        raise ImportError(
            "MuJoCo engine requires mujoco, mujoco-warp, and warp-lang. "
            "Install them with: pip install \"mujoco>=3.8.0\" "
            "\"mujoco-warp>=3.8.0\" \"warp-lang>=1.12.0\""
        ) from e

    wp.config.enable_backward = False
    _mujoco = mujoco
    _mjwarp = mjwarp
    _mjwarp_support = mjwarp_support
    _wp = wp
    return _mujoco, _mjwarp, _mjwarp_support, _wp


def _build_warp_kernels():
    global _accumulate_contact_kernel

    if (_accumulate_contact_kernel is not None):
        return _accumulate_contact_kernel

    _, _, _, wp = _load_mujoco_modules()

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

    _accumulate_contact_kernel = accumulate_contact_kernel
    return _accumulate_contact_kernel


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
    geom_ids: list[int]
    joint_ids: list[int]
    qpos_ids: list[int]
    qvel_ids: list[int]
    hinge_qpos_ids: list[int]
    hinge_dof_ids: list[int]
    ball_qpos_ids: list[int]
    ball_dof_ids: list[int]
    root_body_id: int
    root_qpos_ids: list[int]
    root_qvel_ids: list[int]
    mocap_id: int | None
    dof_low: np.ndarray
    dof_high: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    torque_lim: np.ndarray
    mass: float


class MujocoEngine(engine.Engine):
    def __init__(self, config, num_envs, device, visualize, record_video=False):
        super().__init__(visualize=visualize)

        self._mujoco, self._mjwarp, self._mjwarp_support, self._wp = _load_mujoco_modules()

        self._device = device
        self._wp_device = self._wp.get_device(device)
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
        self._cmds = []
        self._dof_forces = []

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
        self._apply_start_states()
        self._forward()
        self._build_contact_tensors()

        if (self._visualize_enabled):
            self._build_viewer()

        if (self.enabled_record_video()):
            self._video_recorder = self._build_video_recorder()

        return

    def set_cmd(self, obj_id, cmd):
        if (self._control_mode != engine.ControlMode.none and len(self._cmds) > 0):
            self._cmds[obj_id][:] = cmd
        return

    def step(self):
        if (self.enabled_record_video() and self._recording):
            self._video_recorder.capture_frame()

        self._sync_derived_state()

        for _ in range(self._sim_steps):
            self._apply_cmd()
            with self._wp.ScopedDevice(self._wp_device):
                self._mjwarp.step(self._wp_model, self._wp_data)

        self._state_dirty = True
        self._sync_derived_state()
        self._update_contact_forces()

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
        self._mujoco.mj_forward(self._mj_model, self._viewer_data)
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
        meta = self._obj_metas[obj_id]
        if (meta.mocap_id is not None):
            return self._torch_mocap_pos[:, meta.mocap_id, :]
        elif (len(meta.root_qpos_ids) > 0):
            return self._torch_qpos[:, meta.root_qpos_ids[0]:meta.root_qpos_ids[0] + 3]
        else:
            return self._torch_xpos[:, meta.root_body_id, :]

    def get_root_rot(self, obj_id):
        meta = self._obj_metas[obj_id]
        if (meta.mocap_id is not None):
            quat = self._torch_mocap_quat[:, meta.mocap_id, :]
        elif (len(meta.root_qpos_ids) > 0):
            quat = self._torch_qpos[:, meta.root_qpos_ids[0] + 3:meta.root_qpos_ids[0] + 7]
        else:
            self._sync_derived_state()
            quat = self._torch_xquat[:, meta.root_body_id, :]
        return _quat_wxyz_to_xyzw(quat)

    def get_root_vel(self, obj_id):
        meta = self._obj_metas[obj_id]
        if (len(meta.root_qvel_ids) > 0):
            return self._torch_qvel[:, meta.root_qvel_ids[0]:meta.root_qvel_ids[0] + 3]
        else:
            return self._root_vel_cache[obj_id]

    def get_root_ang_vel(self, obj_id):
        meta = self._obj_metas[obj_id]
        if (len(meta.root_qvel_ids) > 0):
            quat = self.get_root_rot(obj_id)
            ang_vel_b = self._torch_qvel[:, meta.root_qvel_ids[0] + 3:meta.root_qvel_ids[0] + 6]
            return torch_util.quat_rotate(quat, ang_vel_b)
        else:
            return self._root_ang_vel_cache[obj_id]

    def get_dof_pos(self, obj_id):
        return self._read_dof_pos(self._obj_metas[obj_id])

    def get_dof_vel(self, obj_id):
        meta = self._obj_metas[obj_id]
        if (len(meta.qvel_ids) == 0):
            return self._empty_dof_tensor
        return self._torch_qvel[:, meta.qvel_ids]

    def get_dof_forces(self, obj_id):
        return self._dof_forces[obj_id]

    def get_body_pos(self, obj_id):
        self._sync_derived_state()
        meta = self._obj_metas[obj_id]
        return self._torch_xpos[:, meta.body_ids, :]

    def get_body_rot(self, obj_id):
        self._sync_derived_state()
        meta = self._obj_metas[obj_id]
        return _quat_wxyz_to_xyzw(self._torch_xquat[:, meta.body_ids, :])

    def get_body_vel(self, obj_id):
        self._sync_derived_state()
        meta = self._obj_metas[obj_id]
        pos = self._torch_xpos[:, meta.body_ids, :]
        cvel = self._torch_cvel[:, meta.body_ids, :]
        subtree_com = self._torch_subtree_com[:, meta.root_body_id, :].unsqueeze(1)
        vel = _compute_velocity_from_cvel(pos, subtree_com, cvel)
        return vel[..., 0:3]

    def get_body_ang_vel(self, obj_id):
        self._sync_derived_state()
        meta = self._obj_metas[obj_id]
        return self._torch_cvel[:, meta.body_ids, 0:3]

    def get_contact_forces(self, obj_id):
        return self._contact_forces[obj_id]

    def get_ground_contact_forces(self, obj_id):
        return self._ground_contact_forces[obj_id]

    def set_root_pos(self, env_id, obj_id, root_pos):
        meta = self._obj_metas[obj_id]
        root_pos = _as_torch(root_pos, self._device)

        if (meta.mocap_id is not None):
            _assign_env_tensor(self._torch_mocap_pos[:, meta.mocap_id, :], env_id, root_pos)
        elif (len(meta.root_qpos_ids) > 0):
            target = self._torch_qpos[:, meta.root_qpos_ids[0]:meta.root_qpos_ids[0] + 3]
            _assign_env_tensor(target, env_id, root_pos)

        self._state_dirty = True
        return

    def set_root_rot(self, env_id, obj_id, root_rot):
        meta = self._obj_metas[obj_id]
        root_rot = _quat_xyzw_to_wxyz(_as_torch(root_rot, self._device))

        if (meta.mocap_id is not None):
            _assign_env_tensor(self._torch_mocap_quat[:, meta.mocap_id, :], env_id, root_rot)
        elif (len(meta.root_qpos_ids) > 0):
            target = self._torch_qpos[:, meta.root_qpos_ids[0] + 3:meta.root_qpos_ids[0] + 7]
            _assign_env_tensor(target, env_id, root_rot)

        self._state_dirty = True
        return

    def set_root_vel(self, env_id, obj_id, root_vel):
        meta = self._obj_metas[obj_id]
        root_vel = _as_torch(root_vel, self._device)

        if (len(meta.root_qvel_ids) > 0):
            target = self._torch_qvel[:, meta.root_qvel_ids[0]:meta.root_qvel_ids[0] + 3]
            _assign_env_tensor(target, env_id, root_vel)
        else:
            _assign_env_tensor(self._root_vel_cache[obj_id], env_id, root_vel)

        self._state_dirty = True
        return

    def set_root_ang_vel(self, env_id, obj_id, root_ang_vel):
        meta = self._obj_metas[obj_id]
        root_ang_vel = _as_torch(root_ang_vel, self._device)

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
        else:
            _assign_env_tensor(self._root_ang_vel_cache[obj_id], env_id, root_ang_vel)

        self._state_dirty = True
        return

    def set_dof_pos(self, env_id, obj_id, dof_pos):
        self._write_dof_pos(self._obj_metas[obj_id], env_id, _as_torch(dof_pos, self._device))
        self._state_dirty = True
        return

    def set_dof_vel(self, env_id, obj_id, dof_vel):
        meta = self._obj_metas[obj_id]
        dof_vel = _as_torch(dof_vel, self._device)
        if (len(meta.qvel_ids) > 0):
            _assign_indexed_cols(self._torch_qvel, meta.qvel_ids, env_id, dof_vel)
        self._state_dirty = True
        return

    def set_body_pos(self, env_id, obj_id, body_pos):
        return

    def set_body_rot(self, env_id, obj_id, body_rot):
        return

    def set_body_vel(self, env_id, obj_id, body_vel):
        return

    def set_body_ang_vel(self, env_id, obj_id, body_ang_vel):
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
        return self._mujoco.MjData(self._mj_model)

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

        self._mj_data = self._mujoco.MjData(self._mj_model)
        self._mujoco.mj_forward(self._mj_model, self._mj_data)

        with self._wp.ScopedDevice(self._wp_device):
            self._wp_model = self._mjwarp.put_model(self._mj_model)
            if (hasattr(self._wp_model.opt, "ls_parallel")):
                self._wp_model.opt.ls_parallel = True
            self._wp_data = self._mjwarp.put_data(
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
        return self._mujoco.MjSpec.from_string(scene_xml)

    def _create_obj_spec(self, obj_def, color):
        _, file_ext = os.path.splitext(obj_def.asset_file)
        assert(file_ext in [".xml", ".urdf"]), "Unsupported asset format for MuJoCo: {:s}".format(file_ext)

        obj_spec = self._mujoco.MjSpec.from_file(obj_def.asset_file)

        if (obj_def.fix_root):
            obj_spec = self._wrap_fixed_root_mocap(obj_spec)

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

    def _wrap_fixed_root_mocap(self, obj_spec):
        for joint in list(obj_spec.joints):
            if (int(joint.type) == int(self._mujoco.mjtJoint.mjJNT_FREE)):
                obj_spec.delete(joint)

        if (len(obj_spec.bodies) > 1 and obj_spec.bodies[1].mocap):
            return obj_spec

        keyframes = [(np.array(k.qpos), np.array(k.ctrl), k.name) for k in obj_spec.keys]
        for key in list(obj_spec.keys):
            obj_spec.delete(key)

        wrapper_spec = self._mujoco.MjSpec()
        mocap_body = wrapper_spec.worldbody.add_body(name="mocap_base", mocap=True)
        frame = mocap_body.add_frame()
        wrapper_spec.attach(child=obj_spec, prefix="", frame=frame)

        for qpos, ctrl, name in keyframes:
            wrapper_spec.add_key(name=name, qpos=qpos.tolist(), ctrl=ctrl.tolist())
        return wrapper_spec

    def _disable_self_collisions(self, obj_spec):
        bodies = [b for b in obj_spec.bodies[1:] if b.name]
        for i in range(len(bodies)):
            for j in range(i + 1, len(bodies)):
                obj_spec.add_exclude(bodyname1=bodies[i].name, bodyname2=bodies[j].name)
        return

    def _apply_model_options(self):
        integrator_map = {
            "euler": self._mujoco.mjtIntegrator.mjINT_EULER,
            "implicitfast": self._mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        }
        solver_map = {
            "newton": self._mujoco.mjtSolver.mjSOL_NEWTON,
            "cg": self._mujoco.mjtSolver.mjSOL_CG,
            "pgs": self._mujoco.mjtSolver.mjSOL_PGS,
        }
        cone_map = {
            "pyramidal": self._mujoco.mjtCone.mjCONE_PYRAMIDAL,
            "elliptic": self._mujoco.mjtCone.mjCONE_ELLIPTIC,
        }
        jacobian_map = {
            "auto": self._mujoco.mjtJacobian.mjJAC_AUTO,
            "dense": self._mujoco.mjtJacobian.mjJAC_DENSE,
            "sparse": self._mujoco.mjtJacobian.mjJAC_SPARSE,
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
        body_ids = []
        body_names = []
        for body in obj_spec.bodies[1:]:
            body_name = _strip_prefix(body.name, prefix)
            if (body_name == "mocap_base"):
                continue
            body_ids.append(int(body.id))
            body_names.append(body_name)

        if (len(body_ids) == 0):
            body_ids = [1]
            body_names = [obj_def.name]

        root_body_id = body_ids[0]
        mocap_id = int(self._mj_model.body_mocapid[self._mj_model.body_rootid[root_body_id]])
        if (mocap_id < 0):
            mocap_id = None

        geom_ids = [int(geom.id) for geom in obj_spec.geoms]
        joint_ids = [int(joint.id) for joint in obj_spec.joints]
        joint_ids = [j for j in joint_ids
                     if int(self._mj_model.jnt_type[j]) != int(self._mujoco.mjtJoint.mjJNT_FREE)]

        free_joint_ids = [int(joint.id) for joint in obj_spec.joints
                          if int(self._mj_model.jnt_type[int(joint.id)]) == int(self._mujoco.mjtJoint.mjJNT_FREE)]
        if (len(free_joint_ids) > 0):
            free_joint_id = free_joint_ids[0]
            root_qpos_start = int(self._mj_model.jnt_qposadr[free_joint_id])
            root_qvel_start = int(self._mj_model.jnt_dofadr[free_joint_id])
            root_qpos_ids = list(range(root_qpos_start, root_qpos_start + 7))
            root_qvel_ids = list(range(root_qvel_start, root_qvel_start + 6))
        else:
            root_qpos_ids = []
            root_qvel_ids = []

        qpos_ids = []
        qvel_ids = []
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
            qpos_width = _joint_qpos_width(self._mujoco, joint_type)
            dof_width = _joint_dof_width(self._mujoco, joint_type)

            qpos_ids.extend(range(qpos_start, qpos_start + qpos_width))
            qvel_ids.extend(range(qvel_start, qvel_start + dof_width))

            joint_kp = float(self._mj_model.jnt_stiffness[joint_id])
            joint_kd = np.asarray(self._mj_model.dof_damping[qvel_start:qvel_start + dof_width], dtype=np.float32)
            joint_lim = self._get_joint_torque_limit(joint_id, dof_width)

            if (joint_type == int(self._mujoco.mjtJoint.mjJNT_BALL)):
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

        mass = float(np.sum(self._mj_model.body_mass[body_ids]))

        meta = ObjMeta(obj_def=obj_def,
                       body_ids=body_ids,
                       body_names=body_names,
                       geom_ids=geom_ids,
                       joint_ids=joint_ids,
                       qpos_ids=qpos_ids,
                       qvel_ids=qvel_ids,
                       hinge_qpos_ids=hinge_qpos_ids,
                       hinge_dof_ids=hinge_dof_ids,
                       ball_qpos_ids=ball_qpos_ids,
                       ball_dof_ids=ball_dof_ids,
                       root_body_id=root_body_id,
                       root_qpos_ids=root_qpos_ids,
                       root_qvel_ids=root_qvel_ids,
                       mocap_id=mocap_id,
                       dof_low=np.asarray(dof_low, dtype=np.float32),
                       dof_high=np.asarray(dof_high, dtype=np.float32),
                       kp=np.asarray(kp, dtype=np.float32),
                       kd=np.asarray(kd, dtype=np.float32),
                       torque_lim=np.asarray(torque_lim, dtype=np.float32),
                       mass=mass)
        return meta

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

    def _build_sim_tensors(self):
        self._torch_qpos = self._to_torch(self._wp_data.qpos)
        self._torch_qvel = self._to_torch(self._wp_data.qvel)
        self._torch_qfrc_applied = self._to_torch(self._wp_data.qfrc_applied)
        self._torch_xfrc_applied = self._to_torch(self._wp_data.xfrc_applied)
        self._torch_ctrl = self._to_torch(self._wp_data.ctrl)
        self._torch_mocap_pos = self._to_torch(self._wp_data.mocap_pos)
        self._torch_mocap_quat = self._to_torch(self._wp_data.mocap_quat)
        self._torch_xpos = self._to_torch(self._wp_data.xpos)
        self._torch_xquat = self._to_torch(self._wp_data.xquat)
        self._torch_cvel = self._to_torch(self._wp_data.cvel)
        self._torch_subtree_com = self._to_torch(self._wp_data.subtree_com)

        self._empty_dof_tensor = torch.zeros([self.get_num_envs(), 0], device=self._device, dtype=torch.float32)
        self._root_vel_cache = []
        self._root_ang_vel_cache = []
        self._cmds = []
        self._dof_forces = []

        for meta in self._obj_metas:
            num_dofs = len(meta.qvel_ids)
            self._root_vel_cache.append(torch.zeros([self.get_num_envs(), 3], device=self._device, dtype=torch.float32))
            self._root_ang_vel_cache.append(torch.zeros([self.get_num_envs(), 3], device=self._device, dtype=torch.float32))
            self._cmds.append(torch.zeros([self.get_num_envs(), num_dofs], device=self._device, dtype=torch.float32))
            self._dof_forces.append(torch.zeros([self.get_num_envs(), num_dofs], device=self._device, dtype=torch.float32))

        self._torch_qfrc_applied.zero_()
        self._torch_xfrc_applied.zero_()
        if (self._torch_ctrl.numel() > 0):
            self._torch_ctrl.zero_()
        return

    def _to_torch(self, wp_array):
        return self._wp.to_torch(wp_array)

    def _apply_start_states(self):
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

        self._wp_contact_ids = self._wp.array(np.arange(self._wp_data.naconmax), device=self._wp_device, dtype=int)
        self._wp_contact_spatial = self._wp.zeros(self._wp_data.naconmax, device=self._wp_device, dtype=self._wp.spatial_vector)
        self._wp_total_contact_forces = self._wp.zeros((num_objs, self.get_num_envs(), max_bodies, 3),
                                                       device=self._wp_device, dtype=float)
        self._wp_ground_contact_forces = self._wp.zeros((num_objs, self.get_num_envs(), max_bodies, 3),
                                                        device=self._wp_device, dtype=float)

        body_obj_id = np.full(self._mj_model.nbody, -1, dtype=np.int32)
        body_local_id = np.full(self._mj_model.nbody, -1, dtype=np.int32)
        for obj_id, meta in enumerate(self._obj_metas):
            for local_id, body_id in enumerate(meta.body_ids):
                body_obj_id[body_id] = obj_id
                body_local_id[body_id] = local_id

        self._wp_body_obj_id = self._wp.array(body_obj_id, device=self._wp_device, dtype=int)
        self._wp_body_local_id = self._wp.array(body_local_id, device=self._wp_device, dtype=int)

        contact_forces = self._wp.to_torch(self._wp_total_contact_forces)
        ground_forces = self._wp.to_torch(self._wp_ground_contact_forces)

        self._contact_forces = []
        self._ground_contact_forces = []
        for obj_id, meta in enumerate(self._obj_metas):
            num_bodies = len(meta.body_ids)
            self._contact_forces.append(contact_forces[obj_id, :, :num_bodies, :])
            self._ground_contact_forces.append(ground_forces[obj_id, :, :num_bodies, :])

        self._update_contact_forces()
        return

    def _apply_cmd(self):
        self._torch_qfrc_applied.zero_()

        if (self._control_mode == engine.ControlMode.none):
            return

        for obj_id, meta in enumerate(self._obj_metas):
            if (meta.obj_def.disable_motors or len(meta.qvel_ids) == 0):
                continue

            cmd = self._cmds[obj_id]
            dof_vel = self._torch_qvel[:, meta.qvel_ids]
            kp = torch.tensor(meta.kp, device=self._device, dtype=torch.float32)
            kd = torch.tensor(meta.kd, device=self._device, dtype=torch.float32)
            torque_lim = torch.tensor(meta.torque_lim, device=self._device, dtype=torch.float32)

            if (self._control_mode == engine.ControlMode.pos):
                dof_pos = self._read_dof_pos(meta)
                torque = kp * (cmd - dof_pos) - kd * dof_vel
            elif (self._control_mode == engine.ControlMode.vel):
                torque = kd * (cmd - dof_vel)
            elif (self._control_mode == engine.ControlMode.torque):
                torque = cmd
            elif (self._control_mode == engine.ControlMode.pd_explicit):
                dof_pos = self._read_dof_pos(meta)
                torque = kp * (cmd - dof_pos) - kd * dof_vel
            else:
                assert(False), "Unsupported control mode: {}".format(self._control_mode)

            torque = torch.clip(torque, -torque_lim, torque_lim)
            self._torch_qfrc_applied[:, meta.qvel_ids] = torque
            self._dof_forces[obj_id][:] = torque
        return

    def _read_dof_pos(self, meta):
        if (len(meta.qvel_ids) == 0):
            return self._empty_dof_tensor

        out = torch.zeros([self.get_num_envs(), len(meta.qvel_ids)], device=self._device, dtype=torch.float32)

        if (len(meta.hinge_dof_ids) > 0):
            local_ids = [meta.qvel_ids.index(i) for i in meta.hinge_dof_ids]
            out[:, local_ids] = self._torch_qpos[:, meta.hinge_qpos_ids]

        for qpos_start, dof_start in zip(meta.ball_qpos_ids, meta.ball_dof_ids):
            local_start = meta.qvel_ids.index(dof_start)
            quat_xyzw = _quat_wxyz_to_xyzw(self._torch_qpos[:, qpos_start:qpos_start + 4])
            out[:, local_start:local_start + 3] = torch_util.quat_to_exp_map(quat_xyzw)

        return out

    def _write_dof_pos(self, meta, env_id, dof_pos):
        if (len(meta.qvel_ids) == 0):
            return

        if (len(meta.hinge_dof_ids) > 0):
            local_ids = [meta.qvel_ids.index(i) for i in meta.hinge_dof_ids]
            values = dof_pos[..., local_ids]
            _assign_indexed_cols(self._torch_qpos, meta.hinge_qpos_ids, env_id, values)

        for qpos_start, dof_start in zip(meta.ball_qpos_ids, meta.ball_dof_ids):
            local_start = meta.qvel_ids.index(dof_start)
            quat = torch_util.exp_map_to_quat(dof_pos[..., local_start:local_start + 3])
            quat = _quat_xyzw_to_wxyz(quat)
            _assign_env_tensor(self._torch_qpos[:, qpos_start:qpos_start + 4], env_id, quat)
        return

    def _forward(self):
        with self._wp.ScopedDevice(self._wp_device):
            self._mjwarp.forward(self._wp_model, self._wp_data)
        self._state_dirty = False
        return

    def _sync_derived_state(self):
        if (self._state_dirty):
            self._forward()
        return

    def _update_contact_forces(self):
        if (not hasattr(self, "_wp_total_contact_forces")):
            return

        self._wp_total_contact_forces.zero_()
        self._wp_ground_contact_forces.zero_()

        if (self._wp_data.naconmax == 0):
            return

        self._mjwarp_support.contact_force(
            self._wp_model,
            self._wp_data,
            self._wp_contact_ids,
            True,
            self._wp_contact_spatial,
        )

        kernel = _build_warp_kernels()
        self._wp.launch(
            kernel=kernel,
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
        return

    def _build_viewer(self):
        import mujoco.viewer

        self._viewer_data = self._mujoco.MjData(self._mj_model)
        self._viewer_aux_data = self._mujoco.MjData(self._mj_model)
        self._viewer_opt = self._mujoco.MjvOption()
        self._viewer_pert = self._mujoco.MjvPerturb()

        self._copy_env_state_to_mjdata(self._viewer_data, 0, self._get_env_offset(0))
        self._mujoco.mj_forward(self._mj_model, self._viewer_data)

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

        catmask = self._mujoco.mjtCatBit.mjCAT_DYNAMIC.value
        for env_id in range(1, self.get_num_envs()):
            self._copy_env_state_to_mjdata(self._viewer_aux_data, env_id, self._get_env_offset(env_id))
            self._mujoco.mj_forward(self._mj_model, self._viewer_aux_data)
            self._mujoco.mjv_addGeoms(
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

        if (self._mj_model.nmocap > 0):
            target_data.mocap_pos[:] = self._torch_mocap_pos[env_id].detach().cpu().numpy()
            target_data.mocap_quat[:] = self._torch_mocap_quat[env_id].detach().cpu().numpy()
            target_data.mocap_pos[:] += offset

        target_data.xfrc_applied[:] = self._torch_xfrc_applied[env_id].detach().cpu().numpy()
        return

    def _apply_viewer_camera(self):
        if (self._viewer is None):
            return

        cam_vec = self._camera_pos - self._camera_look_at
        distance = max(np.linalg.norm(cam_vec), 1e-5)
        self._viewer.cam.type = self._mujoco.mjtCamera.mjCAMERA_FREE.value
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
                geom.category = self._mujoco.mjtCatBit.mjCAT_DECOR
                self._mujoco.mjv_initGeom(
                    geom=geom,
                    type=self._mujoco.mjtGeom.mjGEOM_LINE.value,
                    size=np.zeros(3, dtype=np.float64),
                    pos=np.zeros(3, dtype=np.float64),
                    mat=np.eye(3, dtype=np.float64).reshape(-1),
                    rgba=rgba,
                )
                self._mujoco.mjv_connector(
                    geom=geom,
                    type=self._mujoco.mjtGeom.mjGEOM_LINE.value,
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

    def _visualize(self):
        return self._viewer is not None

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


def _joint_dof_width(mujoco, joint_type):
    if (joint_type == int(mujoco.mjtJoint.mjJNT_FREE)):
        return 6
    elif (joint_type == int(mujoco.mjtJoint.mjJNT_BALL)):
        return 3
    elif (joint_type == int(mujoco.mjtJoint.mjJNT_HINGE)):
        return 1
    elif (joint_type == int(mujoco.mjtJoint.mjJNT_SLIDE)):
        return 1
    assert(False), "Unsupported MuJoCo joint type: {}".format(joint_type)


def _joint_qpos_width(mujoco, joint_type):
    if (joint_type == int(mujoco.mjtJoint.mjJNT_FREE)):
        return 7
    elif (joint_type == int(mujoco.mjtJoint.mjJNT_BALL)):
        return 4
    elif (joint_type == int(mujoco.mjtJoint.mjJNT_HINGE)):
        return 1
    elif (joint_type == int(mujoco.mjtJoint.mjJNT_SLIDE)):
        return 1
    assert(False), "Unsupported MuJoCo joint type: {}".format(joint_type)


def _quat_wxyz_to_xyzw(q):
    return torch.cat([q[..., 1:4], q[..., 0:1]], dim=-1)


def _quat_xyzw_to_wxyz(q):
    return torch.cat([q[..., 3:4], q[..., 0:3]], dim=-1)


def _compute_velocity_from_cvel(pos, subtree_com, cvel):
    lin_vel_c = cvel[..., 3:6]
    ang_vel_c = cvel[..., 0:3]
    offset = subtree_com - pos
    lin_vel_w = lin_vel_c - torch.cross(ang_vel_c, offset, dim=-1)
    return torch.cat([lin_vel_w, ang_vel_c], dim=-1)


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
