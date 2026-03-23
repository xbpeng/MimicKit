#!/usr/bin/env python3
"""
Export any MimicKit MJCF character + pretrained model to JSON for the web demo.

Usage:
    python tools/export_humanoid_json.py \\
        --mjcf data/assets/sword_shield/humanoid_sword_shield.xml \\
        --model data/models/ase_humanoid_sword_shield_model.pt \\
        --output web/humanoid_data.json \\
        --pelvis_z 0.903 \\
        --fixed_bodies sword,shield,left_hand

    # Minimal (just MJCF, no model — skips normalizer export):
    python tools/export_humanoid_json.py \\
        --mjcf data/assets/humanoid/humanoid.xml
"""

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Defaults for the sword_shield humanoid
_DEFAULT_MJCF = os.path.join(REPO_ROOT, "data", "assets", "sword_shield",
                              "humanoid_sword_shield.xml")
_DEFAULT_MODEL = os.path.join(REPO_ROOT, "data", "models",
                               "ase_humanoid_sword_shield_model.pt")
_DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "web", "humanoid_data.json")

# ---------------------------------------------------------------------------
# Math helpers (from mjcf_to_physx_usd.py)
# ---------------------------------------------------------------------------

def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def get_rotation_quat(from_vec, to_vec):
    u = _normalize(np.array(from_vec, dtype=np.float64))
    v = _normalize(np.array(to_vec, dtype=np.float64))
    d = np.dot(u, v)
    if d > 1.0 - 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if d < 1e-6 - 1.0:
        axis = np.cross(np.array([1.0, 0.0, 0.0]), u)
        if np.dot(axis, axis) < 1e-6:
            axis = np.cross(np.array([0.0, 1.0, 0.0]), u)
        axis = _normalize(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0])
    c = np.cross(u, v)
    s = math.sqrt((1.0 + d) * 2.0)
    invs = 1.0 / s
    q = np.array([c[0] * invs, c[1] * invs, c[2] * invs, 0.5 * s])
    return q / np.linalg.norm(q)


def quat_rotate(q, v):
    qv = q[:3]
    qw = q[3]
    v = np.array(v, dtype=np.float64)
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


def mat33_to_quat(cols):
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
        if m11 > m00: i = 1
        if m22 > ([m00, m11, m22][i]): i = 2
        if i == 0:
            h = math.sqrt((m00 - (m11 + m22)) + 1.0)
            x = 0.5 * h; h = 0.5 / h
            y = (m01 + m10) * h; z = (m20 + m02) * h; w = (m21 - m12) * h
        elif i == 1:
            h = math.sqrt((m11 - (m22 + m00)) + 1.0)
            y = 0.5 * h; h = 0.5 / h
            z = (m12 + m21) * h; x = (m01 + m10) * h; w = (m02 - m20) * h
        else:
            h = math.sqrt((m22 - (m00 + m11)) + 1.0)
            z = 0.5 * h; h = 0.5 / h
            x = (m20 + m02) * h; y = (m12 + m21) * h; w = (m10 - m01) * h
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def compute_joint_frame(joint_axes):
    axis_map = [0, 1, 2]
    n = len(joint_axes)
    if n == 0:
        return np.array([0.0, 0.0, 0.0, 1.0]), axis_map
    if n == 1:
        q = get_rotation_quat([1.0, 0.0, 0.0], joint_axes[0])
        return q, axis_map
    if n == 2:
        Q = get_rotation_quat(joint_axes[0], [1.0, 0.0, 0.0])
        b = _normalize(quat_rotate(Q, np.array(joint_axes[1], dtype=np.float64)))
        if abs(np.dot(b, np.array([0.0, 1.0, 0.0]))) > abs(np.dot(b, np.array([0.0, 0.0, 1.0]))):
            axis_map[1] = 1
            c = _normalize(np.cross(np.array(joint_axes[0], dtype=np.float64),
                                     np.array(joint_axes[1], dtype=np.float64)))
            cols = [_normalize(np.array(joint_axes[0], dtype=np.float64)),
                    _normalize(np.array(joint_axes[1], dtype=np.float64)), c]
            q = mat33_to_quat(cols)
        else:
            axis_map[1] = 2; axis_map[2] = 1
            c = _normalize(np.cross(np.array(joint_axes[1], dtype=np.float64),
                                     np.array(joint_axes[0], dtype=np.float64)))
            cols = [_normalize(np.array(joint_axes[0], dtype=np.float64)),
                    c, _normalize(np.array(joint_axes[1], dtype=np.float64))]
            q = mat33_to_quat(cols)
        return q, axis_map
    if n == 3:
        Q = get_rotation_quat(joint_axes[0], [1.0, 0.0, 0.0])
        b = _normalize(quat_rotate(Q, np.array(joint_axes[1], dtype=np.float64)))
        if abs(np.dot(b, np.array([0.0, 1.0, 0.0]))) > abs(np.dot(b, np.array([0.0, 0.0, 1.0]))):
            axis_map[1] = 1; axis_map[2] = 2
            cols = [_normalize(np.array(joint_axes[0], dtype=np.float64)),
                    _normalize(np.array(joint_axes[1], dtype=np.float64)),
                    _normalize(np.array(joint_axes[2], dtype=np.float64))]
            q = mat33_to_quat(cols)
        else:
            axis_map[1] = 2; axis_map[2] = 1
            cols = [_normalize(np.array(joint_axes[0], dtype=np.float64)),
                    _normalize(np.array(joint_axes[2], dtype=np.float64)),
                    _normalize(np.array(joint_axes[1], dtype=np.float64))]
            q = mat33_to_quat(cols)
        return q, axis_map
    raise ValueError("Cannot handle more than 3 joints per body")


def parse_vec(s, n=3):
    parts = s.strip().split()
    return [float(x) for x in parts[:n]]


# ---------------------------------------------------------------------------
# MJCF Parser
# ---------------------------------------------------------------------------

def parse_mjcf(path):
    tree = ET.parse(path)
    root = tree.getroot()

    # Build actuator map: joint_name -> (gear, frcrange)
    # Only look in the <actuator> section, not defaults
    actuator_map = {}
    actuator_section = root.find("actuator")
    if actuator_section is not None:
        for act in actuator_section.findall("motor"):
            jname = act.get("joint")
            if jname is None:
                continue
            gear = float(act.get("gear", "1"))
            frcrange_str = act.get("actuatorfrcrange")
            if frcrange_str:
                frcrange = parse_vec(frcrange_str, 2)
            else:
                frcrange = [-gear, gear]
            actuator_map[jname] = {"gear": gear, "maxForce": max(abs(frcrange[0]), abs(frcrange[1]))}

    bodies = []
    joints = []
    fixed_joints = []

    # Actuator order (defines DOF order) - only from <actuator> section
    actuator_order = []
    if actuator_section is not None:
        for act in actuator_section.findall("motor"):
            jname = act.get("joint")
            if jname is not None:
                actuator_order.append(jname)

    def process_body(body_elem, parent_name, parent_world_pos):
        name = body_elem.get("name")
        local_pos = parse_vec(body_elem.get("pos", "0 0 0"))
        world_pos = [parent_world_pos[i] + local_pos[i] for i in range(3)]

        # Parse geoms
        geoms = []
        for geom_elem in body_elem.findall("geom"):
            g = {"name": geom_elem.get("name", name)}
            gtype = geom_elem.get("type", "sphere")
            g["type"] = gtype
            if geom_elem.get("pos"):
                g["pos"] = parse_vec(geom_elem.get("pos"))
            else:
                g["pos"] = [0, 0, 0]
            if geom_elem.get("size"):
                g["size"] = parse_vec(geom_elem.get("size"), 10)  # variable length
                # Trim trailing
                size_parts = geom_elem.get("size").strip().split()
                g["size"] = [float(x) for x in size_parts]
            if gtype == "sphere":
                g["radius"] = g["size"][0]
            elif gtype == "capsule":
                g["radius"] = g["size"][0]
                if geom_elem.get("fromto"):
                    ft = parse_vec(geom_elem.get("fromto"), 6)
                    g["fromto"] = ft
            elif gtype == "box":
                g["halfExtents"] = g["size"][:3]
            elif gtype == "cylinder":
                g["radius"] = g["size"][1] if len(g["size"]) > 1 else g["size"][0]
                g["halfHeight"] = g["size"][0] if len(g["size"]) > 1 else 0.1
                if geom_elem.get("fromto"):
                    ft = parse_vec(geom_elem.get("fromto"), 6)
                    g["fromto"] = ft
            if geom_elem.get("density"):
                g["density"] = float(geom_elem.get("density"))
            else:
                g["density"] = 1000
            geoms.append(g)

        body_data = {
            "name": name,
            "parent": parent_name,
            "pos": world_pos,
            "localPos": local_pos,
            "geoms": geoms,
        }
        bodies.append(body_data)

        # Parse joints
        joint_elems = body_elem.findall("joint")
        # Filter out freejoint
        joint_elems = [j for j in joint_elems if j.get("type", "hinge") != "free"
                       and j.tag != "freejoint"]
        # Also check for freejoint tag
        has_freejoint = body_elem.find("freejoint") is not None

        if name in FIXED_JOINT_BODIES:
            fixed_joints.append({
                "name": f"{name}_fixed",
                "parent_body": parent_name,
                "child_body": name,
                "localPos0": local_pos,
            })
        elif joint_elems and parent_name is not None:
            axes_data = []
            joint_axes = []
            for je in joint_elems:
                axis = parse_vec(je.get("axis", "1 0 0"))
                joint_axes.append(axis)
                rng = parse_vec(je.get("range", "-3.14159 3.14159"), 2)
                stiffness = float(je.get("stiffness", "0"))
                damping = float(je.get("damping", "0"))
                armature = float(je.get("armature", "0"))
                jname = je.get("name")
                act_info = actuator_map.get(jname, {"gear": 100, "maxForce": 100})
                axes_data.append({
                    "name": jname,
                    "mjcf_axis": axis,
                    "stiffness": stiffness,
                    "damping": damping,
                    "maxForce": act_info["maxForce"],
                    "range": rng,
                    "armature": armature,
                })

            quat_xyzw, axis_map = compute_joint_frame(joint_axes)
            # Convert to wxyz for output
            local_rot = [float(quat_xyzw[3]), float(quat_xyzw[0]),
                         float(quat_xyzw[1]), float(quat_xyzw[2])]

            joint_data = {
                "name": joint_elems[0].get("name").rsplit("_", 1)[0] if len(joint_elems) > 1 else joint_elems[0].get("name"),
                "parent_body": parent_name,
                "child_body": name,
                "axes": axes_data,
                "axisMap": axis_map[:len(joint_elems)],
                "localPos0": local_pos,
                "localRot": local_rot,
                "jointType": "spherical" if len(joint_elems) > 1 else "revolute",
            }
            joints.append(joint_data)

        # Recurse into child bodies
        for child in body_elem.findall("body"):
            process_body(child, name, world_pos)

    # Find worldbody
    worldbody = root.find("worldbody")
    pelvis = worldbody.find("body")
    pelvis_pos = parse_vec(pelvis.get("pos", "0 0 0"))

    # Process from pelvis
    process_body(pelvis, None, pelvis_pos)

    return bodies, joints, fixed_joints, actuator_order


def auto_detect_fixed_bodies(mjcf_path):
    """Detect bodies with no joints (fixed attachment to parent)."""
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    fixed = set()

    def _scan(body_el, is_root=False):
        if not is_root:
            hinge_joints = [j for j in body_el.findall("joint")
                           if j.get("type", "hinge") in ("hinge", "slide")]
            free_joints = body_el.findall("freejoint")
            if not hinge_joints and not free_joints:
                name = body_el.get("name", "body")
                fixed.add(name)
        for child in body_el.findall("body"):
            _scan(child)

    for body in root.find("worldbody").findall("body"):
        _scan(body, is_root=True)
    return fixed


def main():
    parser = argparse.ArgumentParser(
        description="Export MimicKit MJCF + model to JSON for the web demo")
    parser.add_argument("--mjcf", default=_DEFAULT_MJCF,
                        help="Path to MJCF XML file")
    parser.add_argument("--model", default=_DEFAULT_MODEL,
                        help="Path to .pt model checkpoint (optional)")
    parser.add_argument("--output", default=_DEFAULT_OUTPUT,
                        help="Output JSON path")
    parser.add_argument("--pelvis_z", type=float, default=0.903,
                        help="Root body standing height (world Z)")
    parser.add_argument("--fixed_bodies", default=None,
                        help="Comma-separated body names for fixed joints "
                             "(auto-detected if omitted)")
    parser.add_argument("--latent_dim", type=int, default=64,
                        help="Latent vector dimension for ASE/skill models")
    args = parser.parse_args()

    global FIXED_JOINT_BODIES, PELVIS_Z
    if args.fixed_bodies is not None:
        FIXED_JOINT_BODIES = set(args.fixed_bodies.split(","))
    else:
        FIXED_JOINT_BODIES = auto_detect_fixed_bodies(args.mjcf)
        print(f"Auto-detected fixed bodies: {FIXED_JOINT_BODIES}")
    PELVIS_Z = args.pelvis_z

    print(f"Reading MJCF from {args.mjcf}")
    bodies, joints, fixed_joints, actuator_order = parse_mjcf(args.mjcf)

    print(f"Found {len(bodies)} bodies, {len(joints)} joints, {len(fixed_joints)} fixed joints")
    print(f"Actuator order ({len(actuator_order)} DOFs): {actuator_order}")

    # Build DOF order from actuator order
    dof_info = []
    for act_name in actuator_order:
        for jdata in joints:
            for ai, ax in enumerate(jdata["axes"]):
                if ax["name"] == act_name:
                    phys_axis = jdata["axisMap"][ai]
                    dof_info.append({
                        "joint_name": jdata["name"],
                        "axis_name": act_name,
                        "physx_axis": phys_axis,
                        "child_body": jdata["child_body"],
                    })
                    break

    data = {
        "bodies": bodies,
        "joints": joints,
        "fixedJoints": fixed_joints,
        "actuatorOrder": actuator_order,
        "dofInfo": dof_info,
        "pelvis_z": args.pelvis_z,
        "latent_dim": args.latent_dim,
    }

    # Load model checkpoint for normalizer stats (optional)
    model_path = args.model
    if model_path and os.path.isfile(model_path):
        import torch
        print(f"Loading model from {model_path}")
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

        obs_mean = ckpt["_obs_norm._mean"].numpy().tolist()
        obs_std  = ckpt["_obs_norm._std"].numpy().tolist()
        a_mean   = ckpt["_a_norm._mean"].numpy().tolist()
        a_std    = ckpt["_a_norm._std"].numpy().tolist()

        data["obs_dim"]  = len(obs_mean)
        data["act_dim"]  = len(a_mean)
        data["obs_mean"] = obs_mean
        data["obs_std"]  = obs_std
        data["a_mean"]   = a_mean
        data["a_std"]    = a_std
        print(f"obs_dim={data['obs_dim']}, act_dim={data['act_dim']}")
    else:
        data["obs_dim"] = 0
        data["act_dim"] = len(actuator_order)
        print(f"No model checkpoint — normalizer stats omitted")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
