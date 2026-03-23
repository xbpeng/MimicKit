#!/usr/bin/env python3
"""
MJCF to PhysX USD converter.

Converts a MuJoCo MJCF XML file to a USDA text file with PhysX physics schemas
compatible with ovphysx.  No pxr/usd Python library is used -- the USDA format
is written directly as strings.

This converter matches the behavior of Isaac Lab's C++ MJCF importer:
- computeJointFrame() aligns joint axes to PhysX D6 X-axis
- Single hinge joints use PhysicsRevoluteJoint with axis="X" and rotated localRot
- Multi-hinge joints use PhysicsJoint (D6) with proper axis mapping
- Joint drives use "force" type with maxForce from actuatorfrcrange
- PhysxSchemaPhysxLimitAPI stiffness/damping is applied

Usage:
    python tools/mjcf_to_physx_usd/mjcf_to_physx_usd.py
"""

import math
import os
import xml.etree.ElementTree as ET

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

MJCF_PATH = os.path.join(REPO_ROOT, "data", "assets", "sword_shield",
                          "humanoid_sword_shield.xml")
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "assets", "sword_shield",
                            "humanoid_sword_shield_physx.usda")

# Bodies that are rigidly attached (no MJCF joints) and should get a
# PhysicsFixedJoint.  left_hand is excluded because it has no joints either
# but is handled specially.
FIXED_JOINT_BODIES = {"sword", "shield", "left_hand"}

# ---------------------------------------------------------------------------
# Math helpers (using numpy for quaternion/matrix operations)
# ---------------------------------------------------------------------------

def parse_vec(s, n=3):
    """Parse a whitespace-separated string into a list of n floats."""
    parts = s.strip().split()
    return [float(x) for x in parts[:n]]


def rad_to_deg(r):
    return math.degrees(r)


def vec3_sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def vec3_add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def vec3_scale(a, s):
    return [a[0] * s, a[1] * s, a[2] * s]


def vec3_len(a):
    return math.sqrt(a[0]**2 + a[1]**2 + a[2]**2)


def vec3_normalize(a):
    l = vec3_len(a)
    if l < 1e-12:
        return [0.0, 0.0, 1.0]
    return [x / l for x in a]


def _normalize(v):
    """Normalize a numpy array."""
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def get_rotation_quat(from_vec, to_vec):
    """
    Compute quaternion that rotates from_vec to to_vec.
    Matches Isaac Lab's GetRotationQuat().
    Returns quaternion as (x, y, z, w) -- internal format.
    """
    u = _normalize(np.array(from_vec, dtype=np.float64))
    v = _normalize(np.array(to_vec, dtype=np.float64))
    d = np.dot(u, v)

    if d > 1.0 - 1e-6:
        # Vectors are colinear, return identity
        return np.array([0.0, 0.0, 0.0, 1.0])

    if d < 1e-6 - 1.0:
        # Vectors are opposite, return 180 degree rotation
        axis = np.cross(np.array([1.0, 0.0, 0.0]), u)
        if np.dot(axis, axis) < 1e-6:
            axis = np.cross(np.array([0.0, 1.0, 0.0]), u)
        axis = _normalize(axis)
        # 180 degree rotation: w=0, axis=normalized
        return np.array([axis[0], axis[1], axis[2], 0.0])

    c = np.cross(u, v)
    s = math.sqrt((1.0 + d) * 2.0)
    invs = 1.0 / s
    q = np.array([c[0] * invs, c[1] * invs, c[2] * invs, 0.5 * s])
    return q / np.linalg.norm(q)


def quat_rotate(q, v):
    """
    Rotate vector v by quaternion q. q is (x, y, z, w).
    """
    qv = q[:3]
    qw = q[3]
    v = np.array(v, dtype=np.float64)
    # q * v * q^-1
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


def mat33_to_quat(cols):
    """
    Convert a 3x3 rotation matrix (given as 3 column vectors) to quaternion (x, y, z, w).
    Matches Isaac Lab's XQuat(Matrix33) constructor.
    cols = [col0, col1, col2] where each is a numpy 3-vector.
    Matrix element m(i,j) = cols[j][i]
    """
    m00 = cols[0][0]; m10 = cols[0][1]; m20 = cols[0][2]
    m01 = cols[1][0]; m11 = cols[1][1]; m21 = cols[1][2]
    m02 = cols[2][0]; m12 = cols[2][1]; m22 = cols[2][2]

    tr = m00 + m11 + m22
    if tr >= 0:
        h = math.sqrt(tr + 1.0)
        w = 0.5 * h
        h = 0.5 / h
        x = (m21 - m12) * h
        y = (m02 - m20) * h
        z = (m10 - m01) * h
    else:
        i = 0
        if m11 > m00:
            i = 1
        if m22 > ([m00, m11, m22][i]):
            i = 2

        if i == 0:
            h = math.sqrt((m00 - (m11 + m22)) + 1.0)
            x = 0.5 * h
            h = 0.5 / h
            y = (m01 + m10) * h
            z = (m20 + m02) * h
            w = (m21 - m12) * h
        elif i == 1:
            h = math.sqrt((m11 - (m22 + m00)) + 1.0)
            y = 0.5 * h
            h = 0.5 / h
            z = (m12 + m21) * h
            x = (m01 + m10) * h
            w = (m02 - m20) * h
        else:
            h = math.sqrt((m22 - (m00 + m11)) + 1.0)
            z = 0.5 * h
            h = 0.5 / h
            x = (m20 + m02) * h
            y = (m12 + m21) * h
            w = (m10 - m01) * h

    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def quat_to_wxyz(q_xyzw):
    """Convert (x,y,z,w) quaternion to (w,x,y,z) for USD output."""
    return (q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])


def compute_joint_frame(joint_axes):
    """
    Compute joint frame rotation and axis mapping, matching Isaac Lab's
    computeJointFrame().

    Args:
        joint_axes: list of axis vectors (each a 3-element list), one per joint.

    Returns:
        (quat_xyzw, axis_map) where:
        - quat_xyzw is the joint frame rotation as (x, y, z, w)
        - axis_map is a list of 3 ints mapping joint index -> D6 axis index
          (0=X/twist, 1=Y/swing1, 2=Z/swing2)
    """
    axis_map = [0, 1, 2]
    n = len(joint_axes)

    if n == 0:
        return np.array([0.0, 0.0, 0.0, 1.0]), axis_map

    if n == 1:
        # Align D6 x-axis (1,0,0) with the joint axis
        q = get_rotation_quat([1.0, 0.0, 0.0], joint_axes[0])
        return q, axis_map

    if n == 2:
        # Rotate first joint axis to X-axis
        Q = get_rotation_quat(joint_axes[0], [1.0, 0.0, 0.0])
        b = _normalize(quat_rotate(Q, np.array(joint_axes[1], dtype=np.float64)))

        if abs(np.dot(b, np.array([0.0, 1.0, 0.0]))) > abs(np.dot(b, np.array([0.0, 0.0, 1.0]))):
            axis_map[1] = 1
            c = _normalize(np.cross(np.array(joint_axes[0], dtype=np.float64),
                                     np.array(joint_axes[1], dtype=np.float64)))
            cols = [_normalize(np.array(joint_axes[0], dtype=np.float64)),
                    _normalize(np.array(joint_axes[1], dtype=np.float64)),
                    c]
            q = mat33_to_quat(cols)
        else:
            axis_map[1] = 2
            axis_map[2] = 1
            c = _normalize(np.cross(np.array(joint_axes[1], dtype=np.float64),
                                     np.array(joint_axes[0], dtype=np.float64)))
            cols = [_normalize(np.array(joint_axes[0], dtype=np.float64)),
                    c,
                    _normalize(np.array(joint_axes[1], dtype=np.float64))]
            q = mat33_to_quat(cols)
        return q, axis_map

    if n == 3:
        # Rotate first joint axis to X-axis
        Q = get_rotation_quat(joint_axes[0], [1.0, 0.0, 0.0])
        b = _normalize(quat_rotate(Q, np.array(joint_axes[1], dtype=np.float64)))

        if abs(np.dot(b, np.array([0.0, 1.0, 0.0]))) > abs(np.dot(b, np.array([0.0, 0.0, 1.0]))):
            axis_map[1] = 1
            axis_map[2] = 2
            cols = [_normalize(np.array(joint_axes[0], dtype=np.float64)),
                    _normalize(np.array(joint_axes[1], dtype=np.float64)),
                    _normalize(np.array(joint_axes[2], dtype=np.float64))]
            q = mat33_to_quat(cols)
        else:
            axis_map[1] = 2
            axis_map[2] = 1
            cols = [_normalize(np.array(joint_axes[0], dtype=np.float64)),
                    _normalize(np.array(joint_axes[2], dtype=np.float64)),
                    _normalize(np.array(joint_axes[1], dtype=np.float64))]
            q = mat33_to_quat(cols)
        return q, axis_map

    raise ValueError("Cannot handle more than 3 joints per body")


def rotation_quat_from_two_vecs(v_from, v_to):
    """
    Return a quaternion (w, x, y, z) that rotates unit vector v_from to v_to.
    Used for geom orientation. Returns in USD (w,x,y,z) format.
    """
    v_from = vec3_normalize(v_from)
    v_to = vec3_normalize(v_to)

    dot = sum(a * b for a, b in zip(v_from, v_to))
    dot = max(-1.0, min(1.0, dot))

    if dot > 0.9999999:
        return (1.0, 0.0, 0.0, 0.0)  # identity

    if dot < -0.9999999:
        # 180-degree rotation
        perp = [1.0, 0.0, 0.0]
        if abs(v_from[0]) > 0.9:
            perp = [0.0, 1.0, 0.0]
        ax = [
            v_from[1] * perp[2] - v_from[2] * perp[1],
            v_from[2] * perp[0] - v_from[0] * perp[2],
            v_from[0] * perp[1] - v_from[1] * perp[0],
        ]
        ax = vec3_normalize(ax)
        return (0.0, ax[0], ax[1], ax[2])

    cross = [
        v_from[1] * v_to[2] - v_from[2] * v_to[1],
        v_from[2] * v_to[0] - v_from[0] * v_to[2],
        v_from[0] * v_to[1] - v_from[1] * v_to[0],
    ]
    w = 1.0 + dot
    length = math.sqrt(w**2 + cross[0]**2 + cross[1]**2 + cross[2]**2)
    w /= length
    cross = [x / length for x in cross]
    return (w, cross[0], cross[1], cross[2])


def principal_axis_from_vec(axis_vec):
    """Return the closest principal axis label ('X', 'Y', 'Z') given a 3-vector."""
    ax, ay, az = [abs(x) for x in axis_vec]
    if ax >= ay and ax >= az:
        return "X"
    if ay >= ax and ay >= az:
        return "Y"
    return "Z"


def fmt_vec3(v):
    return "({:.6g}, {:.6g}, {:.6g})".format(*v)


def fmt_quatf(q):
    """q = (w, x, y, z) for USD output"""
    return "({:.6g}, {:.6g}, {:.6g}, {:.6g})".format(*q)


# ---------------------------------------------------------------------------
# MJCF default-class inheritance
# ---------------------------------------------------------------------------

class MJCFDefaults:
    """Stores resolved default attributes for geom and joint."""

    def __init__(self):
        self.geom = {
            "condim": "1",
            "friction": "1.0 0.05 0.05",
            "density": "1000",
        }
        self.joint = {
            "limited": "true",
            "stiffness": "0",
            "damping": "0",
            "armature": "0",
        }

    def apply_xml_defaults(self, root):
        default_el = root.find("default")
        if default_el is None:
            return
        for child in default_el:
            if child.tag == "geom":
                self.geom.update(child.attrib)
            elif child.tag == "joint":
                self.joint.update(child.attrib)
            elif child.tag == "default":
                for grandchild in child:
                    if grandchild.tag == "geom":
                        self.geom.update(grandchild.attrib)
                    elif grandchild.tag == "joint":
                        self.joint.update(grandchild.attrib)

    def get_geom(self, attrib):
        merged = dict(self.geom)
        merged.update(attrib)
        return merged

    def get_joint(self, attrib):
        merged = dict(self.joint)
        merged.update(attrib)
        return merged


# ---------------------------------------------------------------------------
# USD text writer helpers
# ---------------------------------------------------------------------------

class IndentWriter:
    """Simple indented line writer."""

    def __init__(self):
        self._lines = []
        self._indent = 0

    def indent(self, n=1):
        self._indent += n

    def dedent(self, n=1):
        self._indent -= n

    def line(self, text=""):
        if text:
            self._lines.append("    " * self._indent + text)
        else:
            self._lines.append("")

    def text(self):
        return "\n".join(self._lines) + "\n"


# ---------------------------------------------------------------------------
# Body info collected in DFS pass
# ---------------------------------------------------------------------------

class BodyInfo:
    """All data needed for one MJCF body to write USD prims."""

    def __init__(self, name, world_pos, local_pos, parent_name,
                 joints_attribs, geom_elements):
        self.name = name
        self.world_pos = world_pos
        self.local_pos = local_pos
        self.parent_name = parent_name
        self.joints_attribs = joints_attribs  # list of resolved joint attr dicts
        self.geom_elements = geom_elements


# ---------------------------------------------------------------------------
# Core converter
# ---------------------------------------------------------------------------

class MJCFToUSDA:

    def __init__(self, mjcf_path, output_path):
        self.mjcf_path = mjcf_path
        self.output_path = output_path
        self.defaults = MJCFDefaults()
        self.w = IndentWriter()

        # Ordered list of BodyInfo (DFS order)
        self.bodies = []
        # Map body_name -> BodyInfo
        self.body_map = {}
        # Map joint_name -> actuator attrib dict
        self.actuator_map = {}

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def convert(self):
        tree = ET.parse(self.mjcf_path)
        root = tree.getroot()

        self.defaults.apply_xml_defaults(root)

        compiler = root.find("compiler")
        self.angle_unit = "radian"
        if compiler is not None:
            self.angle_unit = compiler.get("angle", "radian")

        # Parse actuators
        actuator_el = root.find("actuator")
        if actuator_el is not None:
            for motor in actuator_el.findall("motor"):
                joint_name = motor.get("joint")
                if joint_name:
                    self.actuator_map[joint_name] = motor.attrib
            for pos_act in actuator_el.findall("position"):
                joint_name = pos_act.get("joint")
                if joint_name:
                    self.actuator_map[joint_name] = pos_act.attrib
            for vel_act in actuator_el.findall("velocity"):
                joint_name = vel_act.get("joint")
                if joint_name:
                    self.actuator_map[joint_name] = vel_act.attrib

        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("No <worldbody> found in MJCF")

        # Phase 1: DFS traversal to collect all bodies with world positions
        for body_el in worldbody.findall("body"):
            self._collect_bodies_dfs(body_el,
                                     parent_world_pos=[0.0, 0.0, 0.0],
                                     parent_name=None)

        # Phase 2: Write USD
        self._write_header()

        w = self.w
        w.line('def Xform "World"')
        w.line("{")
        w.indent()

        self._write_physics_scene()
        self._write_ground_plane()

        top_bodies = [b for b in self.bodies if b.parent_name is None]
        for top_body in top_bodies:
            self._write_articulation(top_body)

        w.dedent()
        w.line("}")

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w") as f:
            f.write(w.text())

        print("Wrote:", self.output_path)

    # ------------------------------------------------------------------
    # Phase 1: DFS body collection
    # ------------------------------------------------------------------

    def _collect_bodies_dfs(self, body_el, parent_world_pos, parent_name):
        body_name = body_el.get("name", "body")
        pos_str = body_el.get("pos", "0 0 0")
        local_pos = parse_vec(pos_str)
        world_pos = vec3_add(parent_world_pos, local_pos)

        # Collect hinge joints (not freejoint)
        hinge_joints_raw = [j for j in body_el.findall("joint")
                            if j.get("type", "hinge") == "hinge"]
        hinge_joints_attribs = [self.defaults.get_joint(j.attrib)
                                 for j in hinge_joints_raw]

        geom_elements = list(body_el.findall("geom"))

        info = BodyInfo(
            name=body_name,
            world_pos=world_pos,
            local_pos=local_pos,
            parent_name=parent_name,
            joints_attribs=hinge_joints_attribs,
            geom_elements=geom_elements,
        )
        self.bodies.append(info)
        self.body_map[body_name] = info

        for child_el in body_el.findall("body"):
            self._collect_bodies_dfs(child_el,
                                     parent_world_pos=world_pos,
                                     parent_name=body_name)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def _write_header(self):
        w = self.w
        w.line("#usda 1.0")
        w.line("(")
        w.indent()
        w.line('defaultPrim = "World"')
        w.line("metersPerUnit = 1")
        w.line('upAxis = "Z"')
        w.dedent()
        w.line(")")
        w.line()

    # ------------------------------------------------------------------
    # Physics scene
    # ------------------------------------------------------------------

    def _write_physics_scene(self):
        w = self.w
        w.line('def PhysicsScene "physicsScene"')
        w.line("{")
        w.indent()
        w.line("vector3f physics:gravityDirection = (0, 0, -1)")
        w.line("float physics:gravityMagnitude = 9.81")
        w.dedent()
        w.line("}")
        w.line()

    # ------------------------------------------------------------------
    # Ground plane
    # ------------------------------------------------------------------

    def _write_ground_plane(self):
        w = self.w
        w.line('def Plane "groundPlane" (')
        w.indent()
        w.line('prepend apiSchemas = ["PhysicsCollisionAPI"]')
        w.dedent()
        w.line(")")
        w.line("{")
        w.indent()
        w.line('uniform token axis = "Z"')
        w.line("double3 xformOp:translate = (0, 0, 0)")
        w.line('uniform token[] xformOpOrder = ["xformOp:translate"]')
        w.dedent()
        w.line("}")
        w.line()

    # ------------------------------------------------------------------
    # Articulation root (flat hierarchy)
    # ------------------------------------------------------------------

    def _write_articulation(self, root_body_info):
        w = self.w
        art_name = "humanoid"
        art_usd_path = "/World/{}".format(art_name)

        w.line('def Xform "{}" ('.format(art_name))
        w.indent()
        w.line('prepend apiSchemas = ["PhysicsArticulationRootAPI"]')
        w.dedent()
        w.line(")")
        w.line("{")
        w.indent()

        art_bodies = self._get_subtree(root_body_info.name)

        body_usd_path = {}
        for b in art_bodies:
            safe = self._safe_name(b.name)
            body_usd_path[b.name] = "{}/{}".format(art_usd_path, safe)

        # Pass 1: emit ALL body Xform prims
        for b in art_bodies:
            self._write_body_prim(b, body_usd_path[b.name])

        # Pass 2: emit ALL joint prims
        for b in art_bodies:
            if b.parent_name is None:
                continue

            parent_usd = body_usd_path[b.parent_name]
            child_usd = body_usd_path[b.name]
            local_pos0 = b.local_pos
            local_pos1 = [0.0, 0.0, 0.0]

            if b.joints_attribs:
                self._write_joints_for_body(
                    b.joints_attribs,
                    body_name=b.name,
                    parent_body_usd_path=parent_usd,
                    child_body_usd_path=child_usd,
                    local_pos0=local_pos0,
                    local_pos1=local_pos1,
                )
            else:
                # No joints: emit fixed joint for designated bodies
                if b.name in FIXED_JOINT_BODIES:
                    fixed_name = self._safe_name(b.name) + "_fixed_joint"
                    self._write_fixed_joint(
                        fixed_name,
                        parent_body_usd_path=parent_usd,
                        child_body_usd_path=child_usd,
                        local_pos0=local_pos0,
                        local_pos1=local_pos1,
                    )

        w.dedent()
        w.line("}")
        w.line()

    # ------------------------------------------------------------------
    # Body prim writer (flat, world-space position)
    # ------------------------------------------------------------------

    def _write_body_prim(self, body_info, body_usd_path):
        w = self.w
        safe = self._safe_name(body_info.name)

        w.line('def Xform "{}" ('.format(safe))
        w.indent()
        w.line('prepend apiSchemas = ["PhysicsRigidBodyAPI"]')
        w.dedent()
        w.line(")")
        w.line("{")
        w.indent()

        w.line("double3 xformOp:translate = {}".format(
            fmt_vec3(body_info.world_pos)))
        w.line('uniform token[] xformOpOrder = ["xformOp:translate"]')
        w.line()

        for geom_el in body_info.geom_elements:
            self._write_geom(geom_el, body_usd_path)

        w.dedent()
        w.line("}")
        w.line()

    # ------------------------------------------------------------------
    # Subtree helper
    # ------------------------------------------------------------------

    def _get_subtree(self, root_name):
        result = []
        for b in self.bodies:
            if self._is_in_subtree(b.name, root_name):
                result.append(b)
        return result

    def _is_in_subtree(self, body_name, root_name):
        current = body_name
        while current is not None:
            if current == root_name:
                return True
            info = self.body_map.get(current)
            if info is None:
                return False
            current = info.parent_name
        return False

    # ------------------------------------------------------------------
    # Joint writer: matches Isaac Lab's addJoints + computeJointFrame
    # ------------------------------------------------------------------

    def _write_joints_for_body(self, joints_attribs, body_name,
                               parent_body_usd_path, child_body_usd_path,
                               local_pos0, local_pos1):
        """
        Emit USD joint prims for all joints on a body, matching Isaac Lab's
        behavior exactly.

        - 1 joint: PhysicsRevoluteJoint with axis="X" and rotated localRot
        - 2-3 joints: PhysicsJoint (D6) with proper axis mapping
        """
        # Extract axis vectors for all joints
        joint_axes = []
        for attrib in joints_attribs:
            axis_str = attrib.get("axis", "0 0 1")
            axis_vec = parse_vec(axis_str)
            joint_axes.append(axis_vec)

        # Compute joint frame rotation and axis mapping
        joint_rot_xyzw, axis_map = compute_joint_frame(joint_axes)
        local_rot_wxyz = quat_to_wxyz(joint_rot_xyzw)

        num_joints = len(joints_attribs)

        if num_joints == 1:
            self._write_single_revolute_joint(
                joints_attribs[0],
                parent_body_usd_path=parent_body_usd_path,
                child_body_usd_path=child_body_usd_path,
                local_pos0=local_pos0,
                local_pos1=local_pos1,
                local_rot_wxyz=local_rot_wxyz,
            )
        else:
            # Multi-joint: use D6 joint
            self._write_d6_multi_joint(
                joints_attribs,
                body_name=body_name,
                parent_body_usd_path=parent_body_usd_path,
                child_body_usd_path=child_body_usd_path,
                local_pos0=local_pos0,
                local_pos1=local_pos1,
                local_rot_wxyz=local_rot_wxyz,
                axis_map=axis_map,
            )

    def _write_single_revolute_joint(self, attrib, parent_body_usd_path,
                                     child_body_usd_path, local_pos0,
                                     local_pos1, local_rot_wxyz):
        """
        Emit a PhysicsRevoluteJoint with axis="X" (Isaac Lab convention).
        The joint frame rotation in localRot0/localRot1 aligns the MJCF axis
        with the PhysX X-axis.
        """
        w = self.w
        joint_name = attrib.get("name", "joint")
        safe_jname = self._safe_name(joint_name)

        range_str = attrib.get("range", "-1.5708 1.5708")
        range_parts = range_str.strip().split()
        lower_rad = float(range_parts[0])
        upper_rad = float(range_parts[1])
        if self.angle_unit == "radian":
            lower_deg = rad_to_deg(lower_rad)
            upper_deg = rad_to_deg(upper_rad)
        else:
            lower_deg = lower_rad
            upper_deg = upper_rad

        stiffness = float(attrib.get("stiffness", "0"))
        damping = float(attrib.get("damping", "0"))
        armature = float(attrib.get("armature", "0"))

        # Get maxForce from actuatorfrcrange
        max_force = self._get_max_force(attrib)

        # Build apiSchemas
        api_schemas = ["PhysicsDriveAPI:angular", "PhysxSchemaPhysxJointAPI"]

        w.line('def PhysicsRevoluteJoint "{}" ('.format(safe_jname))
        w.indent()
        w.line('prepend apiSchemas = [{}]'.format(
            ", ".join('"{}"'.format(s) for s in api_schemas)))
        w.dedent()
        w.line(")")
        w.line("{")
        w.indent()

        # Axis is always X after joint frame rotation
        w.line('uniform token physics:axis = "X"')
        w.line("rel physics:body0 = <{}>".format(parent_body_usd_path))
        w.line("rel physics:body1 = <{}>".format(child_body_usd_path))
        w.line("float physics:lowerLimit = {:.4f}".format(lower_deg))
        w.line("float physics:upperLimit = {:.4f}".format(upper_deg))
        w.line("point3f physics:localPos0 = {}".format(fmt_vec3(local_pos0)))
        w.line("point3f physics:localPos1 = {}".format(fmt_vec3(local_pos1)))
        w.line("quatf physics:localRot0 = {}".format(fmt_quatf(local_rot_wxyz)))
        w.line("quatf physics:localRot1 = {}".format(fmt_quatf(local_rot_wxyz)))
        w.line("float physics:breakForce = inf")
        w.line("float physics:breakTorque = inf")

        # PhysxSchemaPhysxJointAPI armature
        w.line("float physxJoint:armature = {:.6g}".format(armature))

        # PhysxSchemaPhysxLimitAPI for X axis
        w.line("float physxLimit:x:physics:stiffness = {:.6g}".format(stiffness))
        w.line("float physxLimit:x:physics:damping = {:.6g}".format(damping))

        # Drive
        w.line("float drive:angular:physics:stiffness = {:.6g}".format(stiffness))
        w.line("float drive:angular:physics:damping = {:.6g}".format(damping))
        w.line("float drive:angular:physics:targetPosition = 0.0")
        w.line('uniform token drive:angular:physics:type = "force"')
        if max_force is not None:
            w.line("float drive:angular:physics:maxForce = {:.6g}".format(max_force))

        w.dedent()
        w.line("}")
        w.line()

    def _write_d6_multi_joint(self, joints_attribs, body_name,
                              parent_body_usd_path, child_body_usd_path,
                              local_pos0, local_pos1, local_rot_wxyz,
                              axis_map):
        """
        Emit a PhysicsJoint (D6) for bodies with 2-3 hinge joints.
        Matches Isaac Lab's behavior: locks translation axes, maps joint DOFs
        to rotX/rotY/rotZ based on axis_map.
        """
        w = self.w
        safe_jname = self._safe_name(body_name) + "_joint"
        num_joints = len(joints_attribs)

        # D6 axis names
        d6_axes = ["transX", "transY", "transZ", "rotX", "rotY", "rotZ"]
        # Hinge axis indices: twist=rotX(3), swing1=rotY(4), swing2=rotZ(5)
        hinge_axis_indices = [3, 4, 5]  # rotX, rotY, rotZ

        # Collect per-joint info with proper axis mapping
        joint_infos = []
        for jid, attrib in enumerate(joints_attribs):
            mapped_d6_idx = hinge_axis_indices[axis_map[jid]]
            d6_name = d6_axes[mapped_d6_idx]

            range_str = attrib.get("range", "-1.5708 1.5708")
            range_parts = range_str.strip().split()
            lower_rad = float(range_parts[0])
            upper_rad = float(range_parts[1])
            if self.angle_unit == "radian":
                lower_deg = rad_to_deg(lower_rad)
                upper_deg = rad_to_deg(upper_rad)
            else:
                lower_deg = lower_rad
                upper_deg = upper_rad

            stiffness = float(attrib.get("stiffness", "0"))
            damping = float(attrib.get("damping", "0"))
            max_force = self._get_max_force(attrib)

            joint_infos.append({
                "d6_name": d6_name,
                "lower_deg": lower_deg,
                "upper_deg": upper_deg,
                "stiffness": stiffness,
                "damping": damping,
                "max_force": max_force,
                "name": attrib.get("name", "joint"),
            })

        # Determine which rotation axes are used and which are locked
        used_rot_axes = set()
        for ji in joint_infos:
            used_rot_axes.add(ji["d6_name"])

        # Determine locked rotation axes
        locked_rot_axes = []
        for rot_ax in ["rotX", "rotY", "rotZ"]:
            if rot_ax not in used_rot_axes:
                locked_rot_axes.append(rot_ax)

        # Build apiSchemas
        api_schemas = []
        for ji in joint_infos:
            api_schemas.append("PhysicsDriveAPI:{}".format(ji["d6_name"]))
        for ji in joint_infos:
            api_schemas.append("PhysicsLimitAPI:{}".format(ji["d6_name"]))
        # Lock translation axes
        for trans_ax in ["transX", "transY", "transZ"]:
            api_schemas.append("PhysicsLimitAPI:{}".format(trans_ax))
        # Lock unused rotation axes
        for rot_ax in locked_rot_axes:
            api_schemas.append("PhysicsLimitAPI:{}".format(rot_ax))
        api_schemas.append("PhysxSchemaPhysxJointAPI")

        armature = float(joints_attribs[0].get("armature", "0"))

        w.line('def PhysicsJoint "{}" ('.format(safe_jname))
        w.indent()
        w.line('prepend apiSchemas = [{}]'.format(
            ", ".join('"{}"'.format(s) for s in api_schemas)))
        w.dedent()
        w.line(")")
        w.line("{")
        w.indent()

        w.line("rel physics:body0 = <{}>".format(parent_body_usd_path))
        w.line("rel physics:body1 = <{}>".format(child_body_usd_path))
        w.line("point3f physics:localPos0 = {}".format(fmt_vec3(local_pos0)))
        w.line("point3f physics:localPos1 = {}".format(fmt_vec3(local_pos1)))
        w.line("quatf physics:localRot0 = {}".format(fmt_quatf(local_rot_wxyz)))
        w.line("quatf physics:localRot1 = {}".format(fmt_quatf(local_rot_wxyz)))
        w.line("float physics:breakForce = inf")
        w.line("float physics:breakTorque = inf")

        # PhysxSchemaPhysxJointAPI armature
        w.line("float physxJoint:armature = {:.6g}".format(armature))
        w.line()

        # Lock translation axes (low > high means locked)
        for trans_ax in ["transX", "transY", "transZ"]:
            w.line("float limit:{}:physics:low = 1".format(trans_ax))
            w.line("float limit:{}:physics:high = -1".format(trans_ax))

        # Lock unused rotation axes
        for rot_ax in locked_rot_axes:
            w.line("float limit:{}:physics:low = 1".format(rot_ax))
            w.line("float limit:{}:physics:high = -1".format(rot_ax))
        w.line()

        # Per-joint limits, drives, and PhysX limit API
        for ji in joint_infos:
            ns = ji["d6_name"]
            w.line("float limit:{}:physics:low = {:.4f}".format(ns, ji["lower_deg"]))
            w.line("float limit:{}:physics:high = {:.4f}".format(ns, ji["upper_deg"]))
            # PhysxSchemaPhysxLimitAPI stiffness/damping
            w.line("float physxLimit:{}:physics:stiffness = {:.6g}".format(ns, ji["stiffness"]))
            w.line("float physxLimit:{}:physics:damping = {:.6g}".format(ns, ji["damping"]))
        w.line()

        for ji in joint_infos:
            ns = ji["d6_name"]
            w.line("float drive:{}:physics:stiffness = {:.6g}".format(ns, ji["stiffness"]))
            w.line("float drive:{}:physics:damping = {:.6g}".format(ns, ji["damping"]))
            w.line("float drive:{}:physics:targetPosition = 0.0".format(ns))
            w.line('uniform token drive:{}:physics:type = "force"'.format(ns))
            if ji["max_force"] is not None:
                w.line("float drive:{}:physics:maxForce = {:.6g}".format(ns, ji["max_force"]))
            w.line()

        w.dedent()
        w.line("}")
        w.line()

    def _get_max_force(self, joint_attrib):
        """
        Get maxForce from actuatorfrcrange (on the joint) or actuator forcerange.
        Matches Isaac Lab's createJointDrives logic.
        """
        joint_name = joint_attrib.get("name", "")

        # Check actuator first
        actuator = self.actuator_map.get(joint_name)
        if actuator is not None:
            # Check actuator forcerange
            forcerange_str = actuator.get("forcerange")
            if forcerange_str:
                parts = forcerange_str.strip().split()
                fr_low = float(parts[0])
                fr_high = float(parts[1])
                max_force = max(abs(fr_low), abs(fr_high))
                if max_force < 1e30:
                    return max_force

        # Fall back to joint's actuatorfrcrange
        frcrange_str = joint_attrib.get("actuatorfrcrange")
        if frcrange_str:
            parts = frcrange_str.strip().split()
            fr_low = float(parts[0])
            fr_high = float(parts[1])
            max_force = max(abs(fr_low), abs(fr_high))
            if max_force < 1e30:
                return max_force

        return None

    # ------------------------------------------------------------------
    # Fixed joint writer
    # ------------------------------------------------------------------

    def _write_fixed_joint(self, joint_prim_name, parent_body_usd_path,
                           child_body_usd_path, local_pos0, local_pos1):
        w = self.w
        safe_jname = self._safe_name(joint_prim_name)

        w.line('def PhysicsFixedJoint "{}"'.format(safe_jname))
        w.line("{")
        w.indent()
        w.line("rel physics:body0 = <{}>".format(parent_body_usd_path))
        w.line("rel physics:body1 = <{}>".format(child_body_usd_path))
        w.line("point3f physics:localPos0 = {}".format(fmt_vec3(local_pos0)))
        w.line("point3f physics:localPos1 = {}".format(fmt_vec3(local_pos1)))
        w.line("quatf physics:localRot0 = (1, 0, 0, 0)")
        w.line("quatf physics:localRot1 = (1, 0, 0, 0)")
        w.dedent()
        w.line("}")
        w.line()

    # ------------------------------------------------------------------
    # Geom writer
    # ------------------------------------------------------------------

    def _write_geom(self, geom_el, body_usd_path):
        attrib = self.defaults.get_geom(geom_el.attrib)
        geom_type = attrib.get("type", "sphere")
        geom_name = attrib.get("name", "geom")
        safe_name = self._safe_name(geom_name)
        density = float(attrib.get("density", "1000"))

        api_schemas = ["PhysicsCollisionAPI", "PhysicsMassAPI"]

        if geom_type == "sphere":
            self._write_sphere_geom(attrib, safe_name, density, api_schemas)
        elif geom_type == "capsule":
            self._write_capsule_geom(attrib, safe_name, density, api_schemas)
        elif geom_type == "box":
            self._write_box_geom(attrib, safe_name, density, api_schemas)
        elif geom_type == "cylinder":
            self._write_cylinder_geom(attrib, safe_name, density, api_schemas)
        else:
            self._write_sphere_geom(attrib, safe_name, density, api_schemas)

    def _write_sphere_geom(self, attrib, safe_name, density, api_schemas):
        w = self.w
        size_str = attrib.get("size", "0.05")
        radius = float(size_str.strip().split()[0])
        pos_str = attrib.get("pos", "0 0 0")
        pos = parse_vec(pos_str)

        w.line('def Sphere "{}" ('.format(safe_name))
        w.indent()
        w.line('prepend apiSchemas = ["{}"]'.format('", "'.join(api_schemas)))
        w.dedent()
        w.line(")")
        w.line("{")
        w.indent()
        w.line("double radius = {:.6g}".format(radius))
        w.line("float physics:density = {:.4g}".format(density))
        w.line("double3 xformOp:translate = {}".format(fmt_vec3(pos)))
        w.line('uniform token[] xformOpOrder = ["xformOp:translate"]')
        w.dedent()
        w.line("}")
        w.line()

    def _write_capsule_geom(self, attrib, safe_name, density, api_schemas):
        w = self.w
        size_str = attrib.get("size", "0.05")
        size_parts = size_str.strip().split()
        radius = float(size_parts[0])

        fromto_str = attrib.get("fromto", None)
        if fromto_str is not None:
            pts = parse_vec(fromto_str, 6)
            p1 = pts[0:3]
            p2 = pts[3:6]
            center = vec3_scale(vec3_add(p1, p2), 0.5)
            diff = vec3_sub(p2, p1)
            length = vec3_len(diff)
            half_height = length / 2.0
            direction = vec3_normalize(diff)
        else:
            pos_str = attrib.get("pos", "0 0 0")
            center = parse_vec(pos_str)
            half_height = float(size_parts[1]) if len(size_parts) > 1 else radius
            direction = [0.0, 0.0, 1.0]

        axis_label = principal_axis_from_vec(direction)

        default_axis_map = {"X": [1., 0., 0.], "Y": [0., 1., 0.], "Z": [0., 0., 1.]}
        usd_default = default_axis_map[axis_label]
        quat = rotation_quat_from_two_vecs(usd_default, direction)
        identity_quat = (1.0, 0.0, 0.0, 0.0)
        is_identity = all(abs(quat[i] - identity_quat[i]) < 1e-5 for i in range(4))

        w.line('def Capsule "{}" ('.format(safe_name))
        w.indent()
        w.line('prepend apiSchemas = ["{}"]'.format('", "'.join(api_schemas)))
        w.dedent()
        w.line(")")
        w.line("{")
        w.indent()
        w.line('uniform token axis = "{}"'.format(axis_label))
        w.line("double height = {:.6g}".format(half_height * 2.0))
        w.line("double radius = {:.6g}".format(radius))
        w.line("float physics:density = {:.4g}".format(density))

        xform_ops = ["xformOp:translate"]
        w.line("double3 xformOp:translate = {}".format(fmt_vec3(center)))
        if not is_identity:
            w.line("quatf xformOp:orient = {}".format(fmt_quatf(quat)))
            xform_ops.append("xformOp:orient")
        w.line('uniform token[] xformOpOrder = [{}]'.format(
            ", ".join('"{}"'.format(op) for op in xform_ops)))
        w.dedent()
        w.line("}")
        w.line()

    def _write_box_geom(self, attrib, safe_name, density, api_schemas):
        w = self.w
        size_str = attrib.get("size", "0.1 0.1 0.1")
        size_parts = [float(x) for x in size_str.strip().split()]
        hx = size_parts[0] if len(size_parts) > 0 else 0.1
        hy = size_parts[1] if len(size_parts) > 1 else hx
        hz = size_parts[2] if len(size_parts) > 2 else hx
        pos_str = attrib.get("pos", "0 0 0")
        pos = parse_vec(pos_str)

        w.line('def Cube "{}" ('.format(safe_name))
        w.indent()
        w.line('prepend apiSchemas = ["{}"]'.format('", "'.join(api_schemas)))
        w.dedent()
        w.line(")")
        w.line("{")
        w.indent()
        w.line("double size = 1")
        w.line("float physics:density = {:.4g}".format(density))
        w.line("double3 xformOp:translate = {}".format(fmt_vec3(pos)))
        w.line("double3 xformOp:scale = ({:.6g}, {:.6g}, {:.6g})".format(
            hx * 2, hy * 2, hz * 2))
        w.line('uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]')
        w.dedent()
        w.line("}")
        w.line()

    def _write_cylinder_geom(self, attrib, safe_name, density, api_schemas):
        w = self.w
        size_str = attrib.get("size", "0.1")
        size_parts = size_str.strip().split()
        radius = float(size_parts[0])
        fromto_str = attrib.get("fromto", None)
        pos_str = attrib.get("pos", "0 0 0")

        if fromto_str is not None:
            pts = parse_vec(fromto_str, 6)
            p1, p2 = pts[0:3], pts[3:6]
            center = vec3_scale(vec3_add(p1, p2), 0.5)
            diff = vec3_sub(p2, p1)
            height = vec3_len(diff)
            direction = vec3_normalize(diff)
        else:
            center = parse_vec(pos_str)
            height = float(size_parts[1]) * 2 if len(size_parts) > 1 else 0.1
            direction = [0.0, 0.0, 1.0]

        axis_label = principal_axis_from_vec(direction)
        default_axis_map = {"X": [1., 0., 0.], "Y": [0., 1., 0.], "Z": [0., 0., 1.]}
        usd_default = default_axis_map[axis_label]
        quat = rotation_quat_from_two_vecs(usd_default, direction)
        identity_quat = (1.0, 0.0, 0.0, 0.0)
        is_identity = all(abs(quat[i] - identity_quat[i]) < 1e-5 for i in range(4))

        w.line('def Cylinder "{}" ('.format(safe_name))
        w.indent()
        w.line('prepend apiSchemas = ["{}"]'.format('", "'.join(api_schemas)))
        w.dedent()
        w.line(")")
        w.line("{")
        w.indent()
        w.line('uniform token axis = "{}"'.format(axis_label))
        w.line("double height = {:.6g}".format(height))
        w.line("double radius = {:.6g}".format(radius))
        w.line("float physics:density = {:.4g}".format(density))

        xform_ops = ["xformOp:translate"]
        w.line("double3 xformOp:translate = {}".format(fmt_vec3(center)))
        if not is_identity:
            w.line("quatf xformOp:orient = {}".format(fmt_quatf(quat)))
            xform_ops.append("xformOp:orient")
        w.line('uniform token[] xformOpOrder = [{}]'.format(
            ", ".join('"{}"'.format(op) for op in xform_ops)))
        w.dedent()
        w.line("}")
        w.line()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_name(name):
        """Convert an MJCF name to a valid USD prim name."""
        return name.replace(" ", "_").replace("-", "_").replace(".", "_")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    converter = MJCFToUSDA(MJCF_PATH, OUTPUT_PATH)
    converter.convert()
