#!/usr/bin/env python3
"""
MJCF to PhysX USD converter.

Converts a MuJoCo MJCF XML file to a USDA text file with PhysX physics schemas
compatible with ovphysx.  No pxr/usd Python library is used – the USDA format
is written directly as strings.

Architecture: FLAT USD hierarchy where ALL body prims and ALL joint prims are
DIRECT children of the articulation root Xform, with WORLD-SPACE transforms.
No nested body Xforms.  This is how Isaac Lab and professional PhysX setups work.

Usage:
    python tools/mjcf_to_physx_usd/mjcf_to_physx_usd.py
"""

import math
import os
import xml.etree.ElementTree as ET

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
# PhysicsFixedJoint.  left_hand is excluded because it has left_hand_x/y/z joints.
FIXED_JOINT_BODIES = {"sword", "shield"}

# ---------------------------------------------------------------------------
# Math helpers
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


def rotation_quat_from_two_vecs(v_from, v_to):
    """
    Return a quaternion (w, x, y, z) that rotates unit vector v_from to v_to.
    Both inputs must already be unit vectors.
    """
    v_from = vec3_normalize(v_from)
    v_to = vec3_normalize(v_to)

    dot = sum(a * b for a, b in zip(v_from, v_to))
    dot = max(-1.0, min(1.0, dot))

    if dot > 0.9999999:
        return (1.0, 0.0, 0.0, 0.0)  # identity

    if dot < -0.9999999:
        # 180-degree rotation – pick an orthogonal axis
        perp = [1.0, 0.0, 0.0]
        if abs(v_from[0]) > 0.9:
            perp = [0.0, 1.0, 0.0]
        # cross product
        ax = [
            v_from[1] * perp[2] - v_from[2] * perp[1],
            v_from[2] * perp[0] - v_from[0] * perp[2],
            v_from[0] * perp[1] - v_from[1] * perp[0],
        ]
        ax = vec3_normalize(ax)
        return (0.0, ax[0], ax[1], ax[2])

    # general case
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
    """
    Return the closest principal axis label ('X', 'Y', 'Z') given a 3-vector.
    """
    ax, ay, az = [abs(x) for x in axis_vec]
    if ax >= ay and ax >= az:
        return "X"
    if ay >= ax and ay >= az:
        return "Y"
    return "Z"


def axis_vec_to_drive_namespace(axis_vec):
    """
    Map an MJCF axis vector to a PhysX drive/limit namespace string.
    (1,0,0) -> 'rotX', (0,1,0) -> 'rotY', (0,0,1) -> 'rotZ'
    """
    label = principal_axis_from_vec(axis_vec)
    return "rot" + label


def fmt_vec3(v):
    return "({:.6g}, {:.6g}, {:.6g})".format(*v)


def fmt_quatf(q):
    """q = (w, x, y, z)"""
    return "({:.6g}, {:.6g}, {:.6g}, {:.6g})".format(*q)


# ---------------------------------------------------------------------------
# MJCF default-class inheritance
# ---------------------------------------------------------------------------

class MJCFDefaults:
    """Stores resolved default attributes for geom and joint."""

    def __init__(self):
        # geom defaults
        self.geom = {
            "condim": "1",
            "friction": "1.0 0.05 0.05",
            "density": "1000",
        }
        # joint defaults
        self.joint = {
            "limited": "true",
            "stiffness": "0",
            "damping": "0",
            "armature": "0",
        }

    def apply_xml_defaults(self, root):
        """Read the <default> section and update our stored defaults."""
        default_el = root.find("default")
        if default_el is None:
            return
        # top-level children that are not class-specific
        for child in default_el:
            if child.tag == "geom":
                self.geom.update(child.attrib)
            elif child.tag == "joint":
                self.joint.update(child.attrib)
            elif child.tag == "default":
                # class-scoped defaults – currently we only handle "body"
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
# Joint grouping helpers
# ---------------------------------------------------------------------------

# Suffixes used in MJCF to denote individual axes of a multi-DOF joint.
# We strip these to find the "base name" for grouping.
_AXIS_SUFFIXES = ("_x", "_y", "_z")


def _joint_base_name(joint_name):
    """
    Given an MJCF joint name like 'abdomen_x', return the base group name
    'abdomen'.  If the name has no recognised axis suffix the name itself is
    returned unchanged.
    """
    for suf in _AXIS_SUFFIXES:
        if joint_name.endswith(suf):
            return joint_name[: -len(suf)]
    return joint_name


def group_joints(joints_attribs):
    """
    Given a list of resolved joint attribute dicts (already merged with
    defaults), group them by their base name.

    Returns an ordered list of groups:
        [ (base_name, [attrib_dict, ...]), ... ]

    Single-DOF joints produce groups of length 1.
    Multi-DOF joints (e.g. abdomen_x/y/z) produce groups of length > 1.
    """
    seen = {}    # base_name -> list of attrib dicts (insertion-ordered via list)
    order = []   # keeps insertion order of base_names

    for attrib in joints_attribs:
        name = attrib.get("name", "joint")
        base = _joint_base_name(name)
        if base not in seen:
            seen[base] = []
            order.append(base)
        seen[base].append(attrib)

    return [(b, seen[b]) for b in order]


# ---------------------------------------------------------------------------
# Body info collected in DFS pass
# ---------------------------------------------------------------------------

class BodyInfo:
    """All data needed for one MJCF body to write USD prims."""

    def __init__(self, name, world_pos, local_pos, parent_name,
                 joints_attribs, geom_elements):
        self.name = name                        # MJCF body name
        self.world_pos = world_pos              # [x,y,z] world-space position
        self.local_pos = local_pos              # [x,y,z] pos relative to parent body
        self.parent_name = parent_name          # parent body name or None
        self.joints_attribs = joints_attribs    # list of resolved joint attr dicts
        self.geom_elements = geom_elements      # list of xml geom elements


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

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def convert(self):
        tree = ET.parse(self.mjcf_path)
        root = tree.getroot()

        self.defaults.apply_xml_defaults(root)

        # Check angle units (we default to radians per compiler element)
        compiler = root.find("compiler")
        self.angle_unit = "radian"
        if compiler is not None:
            self.angle_unit = compiler.get("angle", "radian")

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

        # One articulation root per top-level body in worldbody
        # (in practice there is one: pelvis)
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
        """
        Recursively collect BodyInfo for body_el and all descendants.
        Accumulates world positions by summing parent positions.
        """
        body_name = body_el.get("name", "body")
        pos_str = body_el.get("pos", "0 0 0")
        local_pos = parse_vec(pos_str)
        world_pos = vec3_add(parent_world_pos, local_pos)

        # Collect hinge joints (not freejoint)
        hinge_joints_raw = [j for j in body_el.findall("joint")
                            if j.get("type", "hinge") == "hinge"]
        hinge_joints_attribs = [self.defaults.get_joint(j.attrib)
                                 for j in hinge_joints_raw]

        # Collect geom elements
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

        # Recurse into child bodies
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
    # Articulation root  (flat hierarchy)
    # ------------------------------------------------------------------

    def _write_articulation(self, root_body_info):
        """
        Emit the articulation root Xform with ALL bodies and ALL joints
        as DIRECT children (flat hierarchy).  Body prims come first, then
        joint prims.
        """
        w = self.w
        art_name = "humanoid"  # top-level articulation prim name
        art_usd_path = "/World/{}".format(art_name)

        w.line('def Xform "{}" ('.format(art_name))
        w.indent()
        w.line('prepend apiSchemas = ["PhysicsArticulationRootAPI"]')
        w.dedent()
        w.line(")")
        w.line("{")
        w.indent()

        # Determine which bodies belong to this articulation.
        # We include the root body and all descendants.
        art_bodies = self._get_subtree(root_body_info.name)

        # Build USD path map: body_name -> absolute USD path
        # All bodies are direct children of the articulation root.
        body_usd_path = {}
        for b in art_bodies:
            safe = self._safe_name(b.name)
            body_usd_path[b.name] = "{}/{}".format(art_usd_path, safe)

        # ------------------------------------------------------------------
        # Pass 1: emit ALL body Xform prims (with world-space translate)
        # ------------------------------------------------------------------
        for b in art_bodies:
            self._write_body_prim(b, body_usd_path[b.name])

        # ------------------------------------------------------------------
        # Pass 2: emit ALL joint prims
        # ------------------------------------------------------------------
        for b in art_bodies:
            if b.parent_name is None:
                # Root body: no joint connecting it to a parent
                continue

            parent_usd = body_usd_path[b.parent_name]
            child_usd = body_usd_path[b.name]

            # localPos0 = joint anchor in parent body's LOCAL frame
            #           = child body's local translation (MJCF convention)
            local_pos0 = b.local_pos
            local_pos1 = [0.0, 0.0, 0.0]

            if b.joints_attribs:
                groups = group_joints(b.joints_attribs)
                for base_name, group_attribs in groups:
                    joint_prim_name = base_name + "_joint"
                    self._write_joint_group(
                        joint_prim_name, group_attribs,
                        parent_body_usd_path=parent_usd,
                        child_body_usd_path=child_usd,
                        local_pos0=local_pos0,
                        local_pos1=local_pos1,
                    )
            else:
                # No hinge joints: emit fixed joint only for designated bodies
                if b.name in FIXED_JOINT_BODIES:
                    fixed_name = self._safe_name(b.name) + "_fixed_joint"
                    self._write_fixed_joint(
                        fixed_name,
                        parent_body_usd_path=parent_usd,
                        child_body_usd_path=child_usd,
                        local_pos0=local_pos0,
                        local_pos1=local_pos1,
                    )
                # else: body with no joints and not in FIXED_JOINT_BODIES
                # (e.g. left_hand with left_hand_x/y/z handled above)

        w.dedent()
        w.line("}")
        w.line()

    # ------------------------------------------------------------------
    # Body prim writer  (flat, world-space position)
    # ------------------------------------------------------------------

    def _write_body_prim(self, body_info, body_usd_path):
        """
        Emit one body Xform prim as a direct child of the articulation root.
        Uses world-space translate.  Geoms are children with LOCAL positions.
        """
        w = self.w
        safe = self._safe_name(body_info.name)

        w.line('def Xform "{}" ('.format(safe))
        w.indent()
        w.line('prepend apiSchemas = ["PhysicsRigidBodyAPI"]')
        w.dedent()
        w.line(")")
        w.line("{")
        w.indent()

        # World-space position
        w.line("double3 xformOp:translate = {}".format(
            fmt_vec3(body_info.world_pos)))
        w.line('uniform token[] xformOpOrder = ["xformOp:translate"]')
        w.line()

        # Geoms as children (local positions relative to body origin)
        for geom_el in body_info.geom_elements:
            self._write_geom(geom_el, body_usd_path)

        w.dedent()
        w.line("}")
        w.line()

    # ------------------------------------------------------------------
    # Subtree helper
    # ------------------------------------------------------------------

    def _get_subtree(self, root_name):
        """Return ordered list of BodyInfo for root_name and all descendants."""
        result = []
        for b in self.bodies:  # self.bodies is DFS-ordered
            # Check if this body is in the subtree rooted at root_name
            if self._is_in_subtree(b.name, root_name):
                result.append(b)
        return result

    def _is_in_subtree(self, body_name, root_name):
        """Return True if body_name is root_name or a descendant of it."""
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
    # Joint group writer
    # ------------------------------------------------------------------

    def _write_joint_group(self, base_name, group_attribs,
                           parent_body_usd_path, child_body_usd_path,
                           local_pos0, local_pos1):
        """
        Emit ONE USD joint prim for all joints sharing the same body pair.

        - If the group has exactly one joint: emit PhysicsRevoluteJoint with
          PhysicsDriveAPI:angular  (single-axis behaviour).
        - If the group has more than one joint: emit a generic PhysicsJoint
          (D6) with per-axis PhysicsDriveAPI:rotX/Y/Z and PhysicsLimitAPI:rotX/Y/Z.
        """
        w = self.w
        safe_jname = self._safe_name(base_name)

        if len(group_attribs) == 1:
            # ----------------------------------------------------------
            # Single-DOF: use PhysicsRevoluteJoint + PhysicsDriveAPI:angular
            # ----------------------------------------------------------
            attrib = group_attribs[0]
            axis_str = attrib.get("axis", "0 0 1")
            axis_vec = parse_vec(axis_str)
            axis_label = principal_axis_from_vec(axis_vec)

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

            w.line('def PhysicsRevoluteJoint "{}" ('.format(safe_jname))
            w.indent()
            w.line('prepend apiSchemas = ["PhysicsDriveAPI:angular"]')
            w.dedent()
            w.line(")")
            w.line("{")
            w.indent()
            w.line('uniform token physics:axis = "{}"'.format(axis_label))
            w.line("rel physics:body0 = <{}>".format(parent_body_usd_path))
            w.line("rel physics:body1 = <{}>".format(child_body_usd_path))
            w.line("float physics:lowerLimit = {:.4f}".format(lower_deg))
            w.line("float physics:upperLimit = {:.4f}".format(upper_deg))
            w.line("point3f physics:localPos0 = {}".format(fmt_vec3(local_pos0)))
            w.line("point3f physics:localPos1 = {}".format(fmt_vec3(local_pos1)))
            w.line("quatf physics:localRot0 = (1, 0, 0, 0)")
            w.line("quatf physics:localRot1 = (1, 0, 0, 0)")
            w.line("float drive:angular:physics:stiffness = {:.4g}".format(stiffness))
            w.line("float drive:angular:physics:damping = {:.4g}".format(damping))
            w.line("float drive:angular:physics:targetPosition = 0.0")
            w.line('uniform token drive:angular:physics:type = "force"')
            w.dedent()
            w.line("}")
            w.line()

        else:
            # ----------------------------------------------------------
            # Multi-DOF: use generic PhysicsJoint (D6) with per-axis APIs
            # ----------------------------------------------------------
            axes_info = []
            for attrib in group_attribs:
                axis_str = attrib.get("axis", "0 0 1")
                axis_vec = parse_vec(axis_str)
                ns = axis_vec_to_drive_namespace(axis_vec)  # e.g. 'rotX'

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

                axes_info.append((ns, lower_deg, upper_deg, stiffness, damping))

            # Build apiSchemas list
            drive_schemas = ["PhysicsDriveAPI:{}".format(ns)
                             for ns, *_ in axes_info]
            limit_schemas = ["PhysicsLimitAPI:{}".format(ns)
                             for ns, *_ in axes_info]
            all_schemas = drive_schemas + limit_schemas

            w.line('def PhysicsJoint "{}" ('.format(safe_jname))
            w.indent()
            w.line('prepend apiSchemas = [{}]'.format(
                ", ".join('"{}"'.format(s) for s in all_schemas)))
            w.dedent()
            w.line(")")
            w.line("{")
            w.indent()

            w.line("rel physics:body0 = <{}>".format(parent_body_usd_path))
            w.line("rel physics:body1 = <{}>".format(child_body_usd_path))
            w.line("point3f physics:localPos0 = {}".format(fmt_vec3(local_pos0)))
            w.line("point3f physics:localPos1 = {}".format(fmt_vec3(local_pos1)))
            w.line("quatf physics:localRot0 = (1, 0, 0, 0)")
            w.line("quatf physics:localRot1 = (1, 0, 0, 0)")
            w.line()

            for ns, lower_deg, upper_deg, stiffness, damping in axes_info:
                w.line("float drive:{}:physics:stiffness = {:.4g}".format(ns, stiffness))
                w.line("float drive:{}:physics:damping = {:.4g}".format(ns, damping))
                w.line("float drive:{}:physics:targetPosition = 0.0".format(ns))
                w.line('uniform token drive:{}:physics:type = "force"'.format(ns))
                w.line()

            for ns, lower_deg, upper_deg, stiffness, damping in axes_info:
                w.line("float limit:{}:physics:low = {:.4f}".format(ns, lower_deg))
                w.line("float limit:{}:physics:high = {:.4f}".format(ns, upper_deg))

            w.dedent()
            w.line("}")
            w.line()

    # ------------------------------------------------------------------
    # Fixed joint writer  (for bodies with no MJCF <joint> children)
    # ------------------------------------------------------------------

    def _write_fixed_joint(self, joint_prim_name, parent_body_usd_path,
                           child_body_usd_path, local_pos0, local_pos1):
        """
        Emit a PhysicsFixedJoint prim that rigidly connects parent_body to
        child_body.  localPos0 is the child's position in the parent frame;
        localPos1 is the origin.
        """
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

        w = self.w

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
            # Fallback: sphere
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
        """
        Emit a USD Capsule prim.

        MJCF capsules are defined either by:
          - fromto="x1 y1 z1 x2 y2 z2" + size (radius)
          - pos + size (radius halfheight) for axis-aligned capsules
        """
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
            # Axis direction
            direction = vec3_normalize(diff)
        else:
            pos_str = attrib.get("pos", "0 0 0")
            center = parse_vec(pos_str)
            half_height = float(size_parts[1]) if len(size_parts) > 1 else radius
            direction = [0.0, 0.0, 1.0]

        # USD Capsule axis is "X", "Y", or "Z"
        axis_label = principal_axis_from_vec(direction)

        # Compute orientation quaternion from Z-axis (USD default capsule axis)
        # to the actual direction
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
        """Emit a USD Cube prim (half-extents to full size via scale)."""
        w = self.w
        size_str = attrib.get("size", "0.1 0.1 0.1")
        size_parts = [float(x) for x in size_str.strip().split()]
        # MJCF box size = half-extents
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
        # USD Cube has size=2 (unit cube of side 1 by default).
        # We set size=1 and use xformOp:scale to apply half-extents*2.
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
        """Emit a USD Cylinder prim."""
        w = self.w
        size_str = attrib.get("size", "0.1")
        size_parts = size_str.strip().split()
        radius = float(size_parts[0])
        # fromto defines the height axis
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
