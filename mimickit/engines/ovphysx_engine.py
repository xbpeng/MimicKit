import math
import warp as wp
wp.config.enable_backward = False

import newton
import numpy as np
import os
import torch
import xml.etree.ElementTree as ET
import pyglet

import engines.engine as engine

try:
    from ovphysx import (
        PhysX,
        OVPHYSX_TENSOR_ARTICULATION_DOF_POSITION_F32,
        OVPHYSX_TENSOR_ARTICULATION_DOF_VELOCITY_F32,
        OVPHYSX_TENSOR_ARTICULATION_DOF_POSITION_TARGET_F32,
        OVPHYSX_TENSOR_ARTICULATION_LINK_POSE_F32,
        OVPHYSX_TENSOR_ARTICULATION_LINK_VELOCITY_F32,
        OVPHYSX_TENSOR_ARTICULATION_ROOT_POSE_F32,
        OVPHYSX_TENSOR_ARTICULATION_ROOT_VELOCITY_F32,
    )
    _OVPHYSX_AVAILABLE = True
except ImportError:
    _OVPHYSX_AVAILABLE = False


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


def _parse_mjcf_bodies(xml_path):
    """Parse body names from MJCF in depth-first order (tree traversal)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    body_names = []

    def _traverse(node):
        if node.tag == "body":
            name = node.get("name", "")
            body_names.append(name)
            for child in node:
                _traverse(child)

    if worldbody is not None:
        for child in worldbody:
            _traverse(child)

    return body_names


def _parse_mjcf_joints(xml_path):
    """Parse joints from MJCF, returning list of dicts with joint info.

    Collects hinge/slide joints in tree-traversal order, skipping freejoint.
    Returns list of dicts: {name, range_low, range_high, stiffness, damping, actuatorfrcrange_low, actuatorfrcrange_high}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    joints = []

    def _traverse(node):
        if node.tag == "body":
            for child in node:
                if child.tag == "joint":
                    jtype = child.get("type", "hinge")
                    if jtype in ("hinge", "slide"):
                        jrange = child.get("range", "-3.14159 3.14159").split()
                        jstiffness = float(child.get("stiffness", "0"))
                        jdamping = float(child.get("damping", "0"))
                        afrc = child.get("actuatorfrcrange", "0 0").split()
                        joints.append({
                            "name": child.get("name", ""),
                            "range_low": float(jrange[0]),
                            "range_high": float(jrange[1]),
                            "stiffness": jstiffness,
                            "damping": jdamping,
                            "torque_low": float(afrc[0]),
                            "torque_high": float(afrc[1]),
                        })
                elif child.tag == "body":
                    _traverse(child)
            # recurse into sub-bodies
            for child in node:
                if child.tag == "body":
                    _traverse(child)

    if worldbody is not None:
        for child in worldbody:
            _traverse(child)

    return joints


def _parse_mjcf_joints_clean(xml_path):
    """Parse non-free joints from MJCF in DFS order (matching Newton/physics ordering)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    joints = []
    visited_bodies = set()

    def _traverse_body(node):
        body_id = id(node)
        if body_id in visited_bodies:
            return
        visited_bodies.add(body_id)

        # Collect joints belonging to this body (not recursing into sub-bodies here)
        for child in node:
            if child.tag == "joint":
                jtype = child.get("type", "hinge")
                if jtype in ("hinge", "slide"):
                    jrange = child.get("range", "-3.14159 3.14159").split()
                    jstiffness = float(child.get("stiffness", "0"))
                    jdamping = float(child.get("damping", "0"))
                    afrc = child.get("actuatorfrcrange", "0 0").split()
                    joints.append({
                        "name": child.get("name", ""),
                        "range_low": float(jrange[0]),
                        "range_high": float(jrange[1]),
                        "stiffness": jstiffness,
                        "damping": jdamping,
                        "torque_low": float(afrc[0]),
                        "torque_high": float(afrc[1]),
                    })

        # Recurse into child bodies
        for child in node:
            if child.tag == "body":
                _traverse_body(child)

    if worldbody is not None:
        for child in worldbody:
            if child.tag == "body":
                _traverse_body(child)

    return joints


def _parse_mjcf_body_masses(xml_path):
    """Parse per-body total mass from geom density * volume in MJCF."""
    import math

    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    total_mass = 0.0

    def _geom_mass(geom):
        density = float(geom.get("density", "1000"))
        gtype = geom.get("type", "sphere")
        try:
            if gtype == "sphere":
                size = float(geom.get("size", "0.05").split()[0])
                vol = (4.0 / 3.0) * math.pi * size ** 3
            elif gtype == "capsule":
                fromto = geom.get("fromto")
                size_val = float(geom.get("size", "0.05").split()[0])
                if fromto:
                    pts = list(map(float, fromto.split()))
                    p0 = np.array(pts[0:3])
                    p1 = np.array(pts[3:6])
                    L = float(np.linalg.norm(p1 - p0))
                else:
                    L = 0.0
                r = size_val
                vol = math.pi * r * r * L + (4.0 / 3.0) * math.pi * r ** 3
            elif gtype == "cylinder":
                fromto = geom.get("fromto")
                size_parts = geom.get("size", "0.05 0.1").split()
                r = float(size_parts[0])
                if fromto:
                    pts = list(map(float, fromto.split()))
                    p0 = np.array(pts[0:3])
                    p1 = np.array(pts[3:6])
                    h = float(np.linalg.norm(p1 - p0))
                elif len(size_parts) > 1:
                    h = float(size_parts[1]) * 2.0
                else:
                    h = 0.1
                vol = math.pi * r * r * h
            elif gtype == "box":
                size_parts = list(map(float, geom.get("size", "0.05 0.05 0.05").split()))
                vol = 8.0 * size_parts[0] * size_parts[1] * size_parts[2]
            else:
                vol = 0.0
        except Exception:
            vol = 0.0
        return density * vol

    def _traverse(node):
        nonlocal total_mass
        if node.tag == "body":
            for child in node:
                if child.tag == "geom":
                    total_mass += _geom_mass(child)
                _traverse(child)

    if worldbody is not None:
        for child in worldbody:
            _traverse(child)

    return total_mass


class OvPhysXEngine(engine.Engine):
    """Physics engine backend using ovphysx (PhysX 5) for simulation and Newton ViewerGL for visualization.

    Only num_envs=1 is supported — ovphysx does not expose multi-environment batching in this simple binding.
    """

    def __init__(self, config, num_envs, device, visualize, record_video=False):
        super().__init__(visualize=visualize)

        # ovphysx runs on CPU
        self._device = "cpu"

        assert num_envs == 1, "OvPhysXEngine currently supports only num_envs=1"
        self._num_envs = num_envs

        sim_freq = config.get("sim_freq", 120)
        control_freq = config.get("control_freq", 30)
        assert sim_freq >= control_freq and sim_freq % control_freq == 0, \
            "sim_freq must be a positive multiple of control_freq"

        self._sim_freq = sim_freq
        self._control_freq = control_freq
        self._timestep = 1.0 / control_freq
        self._sim_timestep = 1.0 / sim_freq
        self._sim_steps = int(sim_freq / control_freq)
        self._sim_step_count = 0

        self._env_spacing = config.get("env_spacing", 5)

        if "control_mode" in config:
            self._control_mode = engine.ControlMode[config["control_mode"]]
        else:
            self._control_mode = engine.ControlMode.pos

        # Default USD / MJCF files
        self._usd_file = config.get("usd_file", "data/assets/sword_shield/humanoid_sword_shield_physx.usda")

        # Per-env/obj storage
        self._obj_asset_files = []   # list of lists [env_id][obj_id]
        self._obj_types = []
        self._obj_start_pos = []
        self._obj_start_rot = []

        # Will be filled in initialize_sim
        self._physx = None
        self._bindings = {}          # key -> binding object
        self._numpy_cache = {}       # key -> numpy array buffer

        # Visualization (Newton ViewerGL)
        self._vis_model = None
        self._vis_state = None
        self._viewer = None
        self._keyboard_callbacks = {}

        if visualize:
            self._build_viewer()

        # Pending state overrides (applied on next reset)
        self._pending_root_pos = None
        self._pending_root_rot = None
        self._pending_root_vel = None
        self._pending_root_ang_vel = None
        self._pending_dof_pos = None
        self._pending_dof_vel = None

        return

    # ------------------------------------------------------------------
    # Engine interface: identity / metadata
    # ------------------------------------------------------------------

    def get_name(self):
        return "ovphysx"

    def get_num_envs(self):
        return self._num_envs

    def get_timestep(self):
        return self._timestep

    def get_control_mode(self):
        return self._control_mode

    # ------------------------------------------------------------------
    # Engine interface: setup
    # ------------------------------------------------------------------

    def create_env(self):
        env_id = len(self._obj_asset_files)
        self._obj_asset_files.append([])
        self._obj_types.append([])
        self._obj_start_pos.append([])
        self._obj_start_rot.append([])
        return env_id

    def create_obj(self, env_id, obj_type, asset_file, name, is_visual=False,
                   enable_self_collisions=True, fix_root=False,
                   start_pos=None, start_rot=None, color=None, disable_motors=False):
        if start_pos is None:
            start_pos = np.array([0.0, 0.0, 0.0])
        if start_rot is None:
            start_rot = np.array([0.0, 0.0, 0.0, 1.0])

        obj_id = len(self._obj_asset_files[env_id])
        self._obj_asset_files[env_id].append(asset_file)
        self._obj_types[env_id].append(obj_type)
        self._obj_start_pos[env_id].append(np.array(start_pos, dtype=np.float32))
        self._obj_start_rot[env_id].append(np.array(start_rot, dtype=np.float32))
        return obj_id

    def initialize_sim(self):
        assert _OVPHYSX_AVAILABLE, "ovphysx is not installed — cannot use OvPhysXEngine"

        # Resolve USD file from first obj's asset if it ends in .usda, else use config default
        usd_file = self._usd_file
        if (len(self._obj_asset_files) > 0 and len(self._obj_asset_files[0]) > 0):
            candidate = self._obj_asset_files[0][0]
            _, ext = os.path.splitext(candidate)
            if ext in (".usda", ".usd"):
                usd_file = candidate

        # Resolve MJCF asset file (for vis model and metadata parsing)
        mjcf_file = None
        if (len(self._obj_asset_files) > 0 and len(self._obj_asset_files[0]) > 0):
            candidate = self._obj_asset_files[0][0]
            _, ext = os.path.splitext(candidate)
            if ext == ".xml":
                mjcf_file = candidate

        self._usd_file_resolved = usd_file
        self._mjcf_file = mjcf_file

        # ---- Create PhysX simulation ----
        print(f"[OvPhysXEngine] Loading USD: {usd_file}")
        self._physx = PhysX(device="cpu")
        self._usd_handle, _ = self._physx.add_usd(usd_file)
        self._physx.wait_all()

        # ---- Create tensor bindings ----
        artpath = "/World/humanoid"
        self._bind_dof_pos    = self._physx.create_tensor_binding(pattern=artpath, tensor_type=OVPHYSX_TENSOR_ARTICULATION_DOF_POSITION_F32)
        self._bind_dof_vel    = self._physx.create_tensor_binding(pattern=artpath, tensor_type=OVPHYSX_TENSOR_ARTICULATION_DOF_VELOCITY_F32)
        self._bind_dof_target = self._physx.create_tensor_binding(pattern=artpath, tensor_type=OVPHYSX_TENSOR_ARTICULATION_DOF_POSITION_TARGET_F32)
        self._bind_link_pose  = self._physx.create_tensor_binding(pattern=artpath, tensor_type=OVPHYSX_TENSOR_ARTICULATION_LINK_POSE_F32)
        self._bind_link_vel   = self._physx.create_tensor_binding(pattern=artpath, tensor_type=OVPHYSX_TENSOR_ARTICULATION_LINK_VELOCITY_F32)
        self._bind_root_pose  = self._physx.create_tensor_binding(pattern=artpath, tensor_type=OVPHYSX_TENSOR_ARTICULATION_ROOT_POSE_F32)
        self._bind_root_vel   = self._physx.create_tensor_binding(pattern=artpath, tensor_type=OVPHYSX_TENSOR_ARTICULATION_ROOT_VELOCITY_F32)

        # Allocate numpy buffers matching binding shapes
        self._np_dof_pos = np.zeros(self._bind_dof_pos.shape, dtype=np.float32)
        self._np_dof_vel = np.zeros(self._bind_dof_vel.shape, dtype=np.float32)
        self._np_link_pose = np.zeros(self._bind_link_pose.shape, dtype=np.float32)
        self._np_link_vel = np.zeros(self._bind_link_vel.shape, dtype=np.float32)
        self._np_root_pose = np.zeros(self._bind_root_pose.shape, dtype=np.float32)
        self._np_root_vel = np.zeros(self._bind_root_vel.shape, dtype=np.float32)

        # Do an initial read to populate buffers
        self._read_all_bindings()

        # Derive num_dofs / num_links from binding shapes
        # shape is typically (num_art, num_dofs) or (num_dofs,)
        dof_shape = self._bind_dof_pos.shape
        link_shape = self._bind_link_pose.shape

        if len(dof_shape) == 2:
            self._num_dofs = int(dof_shape[1])
        else:
            self._num_dofs = int(dof_shape[0])

        if len(link_shape) == 3:
            self._num_links = int(link_shape[1])
        elif len(link_shape) == 2:
            self._num_links = int(link_shape[0])
        else:
            self._num_links = int(link_shape[0])

        # ---- Build Newton vis model (MJCF only, for ViewerGL) ----
        if self._mjcf_file is not None:
            self._build_vis_model(self._mjcf_file)
        elif self._viewer is not None:
            print("[OvPhysXEngine] Warning: no MJCF file found; visualization body_q sync will be skipped.")

        # ---- Parse MJCF metadata ----
        if self._mjcf_file is not None:
            self._mjcf_joints = _parse_mjcf_joints_clean(self._mjcf_file)
            self._mjcf_body_names = _parse_mjcf_bodies(self._mjcf_file)
        else:
            self._mjcf_joints = []
            self._mjcf_body_names = []

        # Hard-coded mapping: ovphysx link index -> Newton DFS body index
        # Derived from 3D position matching between ovphysx poses and USDA world positions.
        # Newton order: pelvis(0), torso(1), head(2), rua(3), rla(4), rh(5), sword(6),
        #               lua(7), lla(8), shield(9), left_hand(10), rt(11), rs(12), rf(13),
        #               lt(14), ls(15), lf(16)
        self._ovphysx_to_newton = [1, 0, 2, 3, 7, 11, 14, 4, 8, 12, 15, 5, 10, 9, 13, 16, 6]
        # Inverse: newton_body_index -> ovphysx_link_index
        self._newton_to_ovphysx = [0] * 17
        for oi, ni in enumerate(self._ovphysx_to_newton):
            self._newton_to_ovphysx[ni] = oi

        print(f"[OvPhysXEngine] Initialized: {self._num_dofs} DOFs, {self._num_links} links")
        return

    def _build_vis_model(self, mjcf_file):
        """Build a Newton model purely for visualization geometry."""
        builder = newton.ModelBuilder()
        builder.add_mjcf(
            mjcf_file,
            floating=True,
            convert_3d_hinge_to_ball_joints=False,
        )
        self._vis_model = builder.finalize(device="cpu", requires_grad=False)
        self._vis_state = self._vis_model.state()
        # Run FK once so body_q is valid before first render
        newton.eval_fk(self._vis_model, self._vis_state.joint_q, self._vis_state.joint_qd, self._vis_state)

        if self._viewer is not None:
            self._viewer.set_model(self._vis_model)
            self._viewer.set_world_offsets([self._env_spacing, self._env_spacing, 0.0])
        return

    def _build_viewer(self):
        self._viewer = newton.viewer.ViewerGL(headless=False)

        def on_keyboard_event(symbol, modifiers):
            self._on_keyboard_event(symbol, modifiers)

        self._viewer.renderer.register_key_press(on_keyboard_event)
        return

    def _on_keyboard_event(self, symbol, modifiers):
        if symbol in self._keyboard_callbacks:
            self._keyboard_callbacks[symbol]()
        return

    # ------------------------------------------------------------------
    # Engine interface: simulation loop
    # ------------------------------------------------------------------

    def set_cmd(self, obj_id, cmd):
        """Write position targets to the DOF position target binding."""
        # cmd shape: [num_envs, num_dofs] or [num_dofs]
        if isinstance(cmd, torch.Tensor):
            cmd_np = cmd.detach().cpu().numpy().astype(np.float32)
        else:
            cmd_np = np.asarray(cmd, dtype=np.float32)

        # Ensure shape matches binding
        target_buf = np.zeros(self._bind_dof_target.shape, dtype=np.float32)
        flat_cmd = cmd_np.reshape(-1)
        flat_target = target_buf.reshape(-1)
        n = min(len(flat_cmd), len(flat_target))
        flat_target[:n] = flat_cmd[:n]

        # Store for tracking error comparison in diagnostics
        self._last_logged_cmd = flat_cmd.copy() if hasattr(flat_cmd, "copy") else np.array(flat_cmd)
        # PhysX spherical joints require targets in [-pi, pi]
        np.clip(target_buf, -math.pi, math.pi, out=target_buf)
        try:
            self._bind_dof_target.write(target_buf)
        except Exception as e:
            print(f"[OvPhysXEngine] Warning: failed to write DOF targets: {e}")
        return

    def step(self):
        """Run sim_steps PhysX substeps then read back all bindings."""
        for _ in range(self._sim_steps):
            self._physx.step(self._sim_timestep, self._sim_step_count * self._sim_timestep); self._sim_step_count += 1


        try:
            self._physx.wait_all()
        except AttributeError:
            pass  # not all ovphysx versions expose wait_all

        self._read_all_bindings()
        self._sync_vis_state()
        self._log_diagnostics()
        return

    def _read_all_bindings(self):
        self._bind_dof_pos.read(self._np_dof_pos)
        self._bind_dof_vel.read(self._np_dof_vel)
        self._bind_link_pose.read(self._np_link_pose)
        self._bind_link_vel.read(self._np_link_vel)
        self._bind_root_pose.read(self._np_root_pose)
        self._bind_root_vel.read(self._np_root_vel)
        return

    def _sync_vis_state(self):
        """Copy ovphysx link poses into Newton vis state body_q for ViewerGL."""
        if self._vis_state is None:
            return

        # _np_link_pose shape: (num_envs, num_links, 7) or (num_links, 7)
        lp = self._np_link_pose
        if lp.ndim == 3:
            poses = lp[0]  # shape (num_links, 7) for env 0
        elif lp.ndim == 2:
            poses = lp     # shape (num_links, 7)
        else:
            return

        # Newton body_q is a wp.array of wp.transform: shape (N, 7), [px,py,pz,qx,qy,qz,qw]
        # ovphysx pose format is also [px,py,pz,qx,qy,qz,qw] — direct copy
        body_q_np = self._vis_state.body_q.numpy()  # shape (N_newton, 7)

        # Reorder ovphysx poses (ovphysx order) into Newton DFS order
        for oi, ni in enumerate(self._ovphysx_to_newton):
            if ni < len(body_q_np) and oi < len(poses):
                body_q_np[ni] = poses[oi]

        self._vis_state.body_q.assign(wp.array(body_q_np, dtype=wp.transform))
        return

    # ------------------------------------------------------------------
    # Engine interface: visualization / camera
    # ------------------------------------------------------------------

    def set_camera_pose(self, pos, look_at):
        if self._viewer is None:
            return
        import pyglet.math
        dx = float(look_at[0] - pos[0])
        dy = float(look_at[1] - pos[1])
        dz = float(look_at[2] - pos[2])
        pitch = np.arctan2(dz, np.sqrt(dx * dx + dy * dy))
        yaw   = np.arctan2(dy, dx)
        cam_pos = pyglet.math.Vec3(float(pos[0]), float(pos[1]), float(pos[2]))
        self._viewer.set_camera(cam_pos, float(np.rad2deg(pitch)), float(np.rad2deg(yaw)))
        return

    def get_camera_pos(self):
        if self._viewer is None:
            return np.zeros(3, dtype=np.float32)
        p = self._viewer.camera.pos
        return np.array([float(p[0]), float(p[1]), float(p[2])], dtype=np.float32)

    def get_camera_dir(self):
        if self._viewer is None:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        pitch = float(np.deg2rad(self._viewer.camera.pitch))
        yaw   = float(np.deg2rad(self._viewer.camera.yaw))
        d = np.array([np.cos(pitch)*np.cos(yaw), np.cos(pitch)*np.sin(yaw), np.sin(pitch)])
        return d / np.linalg.norm(d)

    def render(self):
        if self._viewer is None:
            return

        sim_time = self._sim_step_count * self._timestep

        self._viewer.end_frame()
        self._viewer.begin_frame(sim_time)
        if self._vis_state is not None:
            self._viewer.log_state(self._vis_state)
        self._draw_ground_grid()

        super().render()
        return

    # ------------------------------------------------------------------
    # Engine interface: state getters
    # ------------------------------------------------------------------


    def _draw_ground_grid(self):
        """Draw a ground grid at z=0 using Newton ViewerGL log_lines."""
        if self._viewer is None:
            return
        size = 5.0    # grid half-extent in metres
        spacing = 1.0 # grid cell size
        import numpy as np_

        lines = int(2 * size / spacing) + 1
        starts, ends, colors = [], [], []
        grey = [0.35, 0.35, 0.35]

        for i in range(lines):
            t = -size + i * spacing
            # line parallel to X axis
            starts.append([- size, t, 0.0])
            ends.append([size, t, 0.0])
            colors.append(grey)
            # line parallel to Y axis
            starts.append([t, -size, 0.0])
            ends.append([t,  size, 0.0])
            colors.append(grey)

        starts = np_.array(starts, dtype=np_.float32)
        ends   = np_.array(ends,   dtype=np_.float32)
        colors = np_.array(colors, dtype=np_.float32)

        self._viewer.log_lines(
            name="ground_grid",
            starts=wp.array(starts, dtype=wp.vec3, device="cpu"),
            ends=wp.array(ends,   dtype=wp.vec3, device="cpu"),
            colors=wp.array(colors, dtype=wp.vec3, device="cpu"),
            width=1.0,
        )

    def _log_diagnostics(self):
        step = self._sim_step_count
        if step % self._sim_steps != 0:
            return
        ctrl = step // self._sim_steps
        if ctrl % 10 != 0:
            return

        root_z     = float(self._np_root_pose.flatten()[2]) if self._np_root_pose.size >= 3 else 0.0
        root_vel   = self._np_root_vel.flatten()
        root_speed = float(np.linalg.norm(root_vel[:3])) if root_vel.size >= 3 else 0.0

        dof_pos = self._np_dof_pos.flatten()
        dof_vel = self._np_dof_vel.flatten()
        max_dof_vel = float(np.max(np.abs(dof_vel))) if dof_vel.size > 0 else 0.0

        cmd = getattr(self, "_last_logged_cmd", None)
        if cmd is not None and dof_pos.size > 0:
            n = min(len(cmd), len(dof_pos))
            tracking_err = float(np.max(np.abs(cmd[:n] - dof_pos[:n])))
        else:
            tracking_err = float("nan")

        dof_range = f"[{float(np.min(dof_pos)):.2f},{float(np.max(dof_pos)):.2f}]" if dof_pos.size > 0 else "[]"
        print(
            f"[OVPHYSX ctrl={ctrl:5d}] root_z={root_z:.3f} spd={root_speed:.3f} "
            f"max_dof_vel={max_dof_vel:.3f} tracking_err={tracking_err:.3f} dof_pos={dof_range}",
            flush=True
        )

    def _root_pose_np(self):
        """Return root pose as (num_envs, 7) numpy array."""
        rp = self._np_root_pose
        if rp.ndim == 1:
            return rp.reshape(1, -1)
        return rp  # (num_envs, 7)

    def _root_vel_np(self):
        """Return root velocity as (num_envs, 6) numpy array [lin(3), ang(3)]."""
        rv = self._np_root_vel
        if rv.ndim == 1:
            return rv.reshape(1, -1)
        return rv  # (num_envs, 6)

    def _dof_pos_np(self):
        dp = self._np_dof_pos
        if dp.ndim == 1:
            return dp.reshape(1, -1)
        return dp

    def _dof_vel_np(self):
        dv = self._np_dof_vel
        if dv.ndim == 1:
            return dv.reshape(1, -1)
        return dv

    def _link_pose_np(self):
        """Return link poses reordered into Newton DFS body order (num_envs, 17, 7)."""
        lp = self._np_link_pose
        if lp.ndim == 2:
            lp = lp.reshape(1, lp.shape[0], lp.shape[1])
        num_envs, num_links, feat = lp.shape
        n_newton = len(self._ovphysx_to_newton)
        out = np.zeros((num_envs, n_newton, feat), dtype=np.float32)
        for oi, ni in enumerate(self._ovphysx_to_newton):
            if oi < num_links:
                out[:, ni, :] = lp[:, oi, :]
        return out

    def _link_vel_np(self):
        """Return link velocities reordered into Newton DFS body order (num_envs, 17, 6)."""
        lv = self._np_link_vel
        if lv.ndim == 2:
            lv = lv.reshape(1, lv.shape[0], lv.shape[1])
        num_envs, num_links, feat = lv.shape
        n_newton = len(self._ovphysx_to_newton)
        out = np.zeros((num_envs, n_newton, feat), dtype=np.float32)
        for oi, ni in enumerate(self._ovphysx_to_newton):
            if oi < num_links:
                out[:, ni, :] = lv[:, oi, :]
        return out

    def get_root_pos(self, obj_id):
        # Returns (num_envs, 3) torch tensor
        rp = self._root_pose_np()
        return torch.tensor(rp[:, :3], dtype=torch.float32)

    def get_root_rot(self, obj_id):
        # Returns (num_envs, 4) torch tensor [qx,qy,qz,qw]
        rp = self._root_pose_np()
        return torch.tensor(rp[:, 3:7], dtype=torch.float32)

    def get_root_vel(self, obj_id):
        # Returns (num_envs, 3) torch tensor — linear velocity
        rv = self._root_vel_np()
        return torch.tensor(rv[:, :3], dtype=torch.float32)

    def get_root_ang_vel(self, obj_id):
        # Returns (num_envs, 3) torch tensor — angular velocity
        rv = self._root_vel_np()
        n = rv.shape[1]
        if n >= 6:
            return torch.tensor(rv[:, 3:6], dtype=torch.float32)
        return torch.zeros((self._num_envs, 3), dtype=torch.float32)

    def get_dof_pos(self, obj_id):
        # Returns (num_envs, num_dofs) torch tensor
        return torch.tensor(self._dof_pos_np(), dtype=torch.float32)

    def get_dof_vel(self, obj_id):
        # Returns (num_envs, num_dofs) torch tensor
        return torch.tensor(self._dof_vel_np(), dtype=torch.float32)

    def get_dof_forces(self, obj_id):
        # ovphysx does not expose DOF forces yet — return zeros
        return torch.zeros((self._num_envs, self._num_dofs), dtype=torch.float32)

    def get_body_pos(self, obj_id):
        # Returns (num_envs, num_links, 3) torch tensor
        lp = self._link_pose_np()
        return torch.tensor(lp[:, :, :3], dtype=torch.float32)

    def get_body_rot(self, obj_id):
        # Returns (num_envs, num_links, 4) torch tensor [qx,qy,qz,qw]
        lp = self._link_pose_np()
        return torch.tensor(lp[:, :, 3:7], dtype=torch.float32)

    def get_body_vel(self, obj_id):
        # Returns (num_envs, num_links, 3) torch tensor — linear velocity
        lv = self._link_vel_np()
        return torch.tensor(lv[:, :, :3], dtype=torch.float32)

    def get_body_ang_vel(self, obj_id):
        # Returns (num_envs, num_links, 3) torch tensor — angular velocity
        lv = self._link_vel_np()
        n = lv.shape[2]
        if n >= 6:
            return torch.tensor(lv[:, :, 3:6], dtype=torch.float32)
        return torch.zeros((self._num_envs, len(self._ovphysx_to_newton), 3), dtype=torch.float32)

    def get_contact_forces(self, obj_id):
        # ovphysx does not expose per-body contact forces yet — return zeros
        return torch.zeros((self._num_envs, len(self._ovphysx_to_newton), 3), dtype=torch.float32)

    def get_ground_contact_forces(self, obj_id):
        # return zeros
        return torch.zeros((self._num_envs, len(self._ovphysx_to_newton), 3), dtype=torch.float32)

    # ------------------------------------------------------------------
    # Engine interface: state setters
    # Store pending overrides; applied via physx.reset() on next opportunity.
    # ------------------------------------------------------------------

    def set_root_pos(self, env_id, obj_id, root_pos):
        if isinstance(root_pos, torch.Tensor):
            root_pos = root_pos.detach().cpu().numpy()
        self._pending_root_pos = np.asarray(root_pos, dtype=np.float32)
        self._apply_pending_state()
        return

    def set_root_rot(self, env_id, obj_id, root_rot):
        if isinstance(root_rot, torch.Tensor):
            root_rot = root_rot.detach().cpu().numpy()
        self._pending_root_rot = np.asarray(root_rot, dtype=np.float32)
        self._apply_pending_state()
        return

    def set_root_vel(self, env_id, obj_id, root_vel):
        if isinstance(root_vel, torch.Tensor):
            root_vel = root_vel.detach().cpu().numpy()
        self._pending_root_vel = np.asarray(root_vel, dtype=np.float32)
        self._apply_pending_state()
        return

    def set_root_ang_vel(self, env_id, obj_id, root_ang_vel):
        if isinstance(root_ang_vel, torch.Tensor):
            root_ang_vel = root_ang_vel.detach().cpu().numpy()
        self._pending_root_ang_vel = np.asarray(root_ang_vel, dtype=np.float32)
        self._apply_pending_state()
        return

    def set_dof_pos(self, env_id, obj_id, dof_pos):
        if isinstance(dof_pos, torch.Tensor):
            dof_pos = dof_pos.detach().cpu().numpy()
        self._pending_dof_pos = np.asarray(dof_pos, dtype=np.float32)
        self._apply_pending_state()
        return

    def set_dof_vel(self, env_id, obj_id, dof_vel):
        if isinstance(dof_vel, torch.Tensor):
            dof_vel = dof_vel.detach().cpu().numpy()
        self._pending_dof_vel = np.asarray(dof_vel, dtype=np.float32)
        self._apply_pending_state()
        return

    def set_body_pos(self, env_id, obj_id, body_pos):
        # No-op: ovphysx does not support per-body position overrides directly
        return

    def set_body_rot(self, env_id, obj_id, body_rot):
        return

    def set_body_vel(self, env_id, obj_id, body_vel):
        return

    def set_body_ang_vel(self, env_id, obj_id, body_ang_vel):
        return

    def set_body_forces(self, env_id, obj_id, body_id, forces):
        return

    def _apply_pending_state(self):
        """Try to write pending state overrides to ovphysx bindings, else call reset()."""
        if self._physx is None:
            return

        applied = False

        # Try writing root pose
        if self._pending_root_pos is not None or self._pending_root_rot is not None:
            try:
                root_pose_buf = np.zeros(self._bind_root_pose.shape, dtype=np.float32)
                self._bind_root_pose.read(root_pose_buf)
                if self._pending_root_pos is not None:
                    pos = self._pending_root_pos.reshape(-1)
                    root_pose_buf.reshape(-1, 7)[:, :3] = pos[:3]
                if self._pending_root_rot is not None:
                    rot = self._pending_root_rot.reshape(-1)
                    root_pose_buf.reshape(-1, 7)[:, 3:7] = rot[:4]
                self._bind_root_pose.write(root_pose_buf)
                self._pending_root_pos = None
                self._pending_root_rot = None
                applied = True
            except Exception:
                pass

        # Try writing root velocity
        if self._pending_root_vel is not None or self._pending_root_ang_vel is not None:
            try:
                root_vel_buf = np.zeros(self._bind_root_vel.shape, dtype=np.float32)
                self._bind_root_vel.read(root_vel_buf)
                if self._pending_root_vel is not None:
                    lin = self._pending_root_vel.reshape(-1)
                    root_vel_buf.reshape(-1, 6)[:, :3] = lin[:3]
                if self._pending_root_ang_vel is not None:
                    ang = self._pending_root_ang_vel.reshape(-1)
                    root_vel_buf.reshape(-1, 6)[:, 3:6] = ang[:3]
                self._bind_root_vel.write(root_vel_buf)
                self._pending_root_vel = None
                self._pending_root_ang_vel = None
                applied = True
            except Exception:
                pass

        # Try writing DOF position
        if self._pending_dof_pos is not None:
            try:
                dof_pos_buf = np.zeros(self._bind_dof_pos.shape, dtype=np.float32)
                flat = self._pending_dof_pos.reshape(-1)
                dof_pos_buf.reshape(-1)[:len(flat)] = flat
                self._bind_dof_pos.write(dof_pos_buf)
                self._pending_dof_pos = None
                applied = True
            except Exception:
                pass

        # Try writing DOF velocity
        if self._pending_dof_vel is not None:
            try:
                dof_vel_buf = np.zeros(self._bind_dof_vel.shape, dtype=np.float32)
                flat = self._pending_dof_vel.reshape(-1)
                dof_vel_buf.reshape(-1)[:len(flat)] = flat
                self._bind_dof_vel.write(dof_vel_buf)
                self._pending_dof_vel = None
                applied = True
            except Exception:
                pass

        # If any pending state couldn't be written, fall back to physx.reset()
        has_pending = any([
            self._pending_root_pos is not None,
            self._pending_root_rot is not None,
            self._pending_root_vel is not None,
            self._pending_root_ang_vel is not None,
            self._pending_dof_pos is not None,
            self._pending_dof_vel is not None,
        ])
        if has_pending:
            self._physx.reset()
            self._pending_root_pos = None
            self._pending_root_rot = None
            self._pending_root_vel = None
            self._pending_root_ang_vel = None
            self._pending_dof_pos = None
            self._pending_dof_vel = None

        return

    # ------------------------------------------------------------------
    # Engine interface: object metadata
    # ------------------------------------------------------------------

    def get_obj_type(self, obj_id):
        return engine.ObjType.articulated

    def get_obj_num_dofs(self, obj_id):
        return self._num_dofs

    def get_obj_num_bodies(self, obj_id):
        return len(self._ovphysx_to_newton)  # Newton body count (17)

    def get_obj_body_names(self, obj_id):
        return list(self._mjcf_body_names)

    def find_obj_body_id(self, obj_id, body_name):
        names = self.get_obj_body_names(obj_id)
        return names.index(body_name)

    def get_obj_torque_limits(self, env_id, obj_id):
        """Return (torque_low, torque_high) arrays parsed from MJCF actuatorfrcrange."""
        if not self._mjcf_joints:
            return (np.zeros(self._num_dofs, dtype=np.float32),
                    np.zeros(self._num_dofs, dtype=np.float32))
        torque_low = np.array([j["torque_low"] for j in self._mjcf_joints], dtype=np.float32)
        torque_high = np.array([j["torque_high"] for j in self._mjcf_joints], dtype=np.float32)
        return torque_low, torque_high

    def get_obj_dof_limits(self, env_id, obj_id):
        """Return (dof_low, dof_high) arrays parsed from MJCF joint range."""
        if not self._mjcf_joints:
            return (np.full(self._num_dofs, -np.pi, dtype=np.float32),
                    np.full(self._num_dofs, np.pi, dtype=np.float32))
        dof_low = np.array([j["range_low"] for j in self._mjcf_joints], dtype=np.float32)
        dof_high = np.array([j["range_high"] for j in self._mjcf_joints], dtype=np.float32)
        return dof_low, dof_high

    def get_obj_pd_gains(self, env_id, obj_id):
        """Return (kp, kd) arrays parsed from MJCF stiffness/damping attributes."""
        if not self._mjcf_joints:
            return (np.zeros(self._num_dofs, dtype=np.float32),
                    np.zeros(self._num_dofs, dtype=np.float32))
        kp = np.array([j["stiffness"] for j in self._mjcf_joints], dtype=np.float32)
        kd = np.array([j["damping"] for j in self._mjcf_joints], dtype=np.float32)
        return kp, kd

    def calc_obj_mass(self, env_id, obj_id):
        """Sum masses from MJCF geometry densities."""
        if self._mjcf_file is None:
            return 0.0
        return _parse_mjcf_body_masses(self._mjcf_file)

    # ------------------------------------------------------------------
    # Engine interface: keyboard callbacks
    # ------------------------------------------------------------------

    def register_keyboard_callback(self, key_str, callback_func):
        if self._viewer is not None:
            key_code = str_to_key_code(key_str)
            assert key_code not in self._keyboard_callbacks
            self._keyboard_callbacks[key_code] = callback_func
        return
