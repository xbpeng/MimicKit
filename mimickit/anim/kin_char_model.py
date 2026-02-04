import abc
import enum
import numpy as np
import torch
import xml.etree.ElementTree as ET

import util.torch_util as torch_util

class JointType(enum.IntEnum):
    ROOT = 0
    HINGE = 1
    SPHERICAL = 2
    FIXED = 3

class Joint():
    def __init__(self, name, joint_type, axis):
        self.name = name
        self.joint_type = joint_type
        self.axis = axis
        self.dof_idx = -1
        return

    def get_dof_dim(self):
        if (self.joint_type == JointType.ROOT):
            dof_dim = 0
        elif (self.joint_type == JointType.HINGE):
            dof_dim = 1
        elif (self.joint_type == JointType.SPHERICAL):
            dof_dim = 3
        elif (self.joint_type == JointType.FIXED):
            dof_dim = 0
        else:
            assert(False), "Unsupported joint type: {:s}".format(self.joint_type)
        return dof_dim

    def get_joint_dof(self, dof):
        # Validate input is a tensor
        if not isinstance(dof, torch.Tensor):
            raise TypeError(
                f"Joint.get_joint_dof() expected torch.Tensor, got {type(dof).__name__}. "
                f"Joint: {self.name}, type: {self.joint_type}"
            )

        dof_idx = self.dof_idx
        dof_dim = self.get_dof_dim()

        # Validate dof_idx was properly initialized
        if dof_idx < 0 and dof_dim > 0:
            raise ValueError(
                f"Joint '{self.name}' has uninitialized dof_idx={dof_idx} but dof_dim={dof_dim}. "
                f"Ensure _label_dof_indices() was called during model initialization."
            )

        j_dof = dof[..., dof_idx:dof_idx + dof_dim]
        return j_dof

    def set_joint_dof(self, j_dof, out_dof):
        dof_idx = self.dof_idx
        dof_dim = self.get_dof_dim()
        out_dof[..., dof_idx:dof_idx + dof_dim] = j_dof
        return

    def dof_to_rot(self, dof):
        # Validate input is a tensor
        if not isinstance(dof, torch.Tensor):
            raise TypeError(
                f"Joint.dof_to_rot() expected torch.Tensor, got {type(dof).__name__}. "
                f"Joint: {self.name}, type: {self.joint_type}, dof_idx: {self.dof_idx}"
            )

        rot_shape = list(dof.shape[:-1])
        rot_shape = rot_shape + [4]
        rot = torch.zeros(rot_shape, device=dof.device, dtype=dof.dtype)

        if (self.joint_type == JointType.ROOT):
            rot[..., -1] = 1
        elif (self.joint_type == JointType.HINGE):
            axis = self.axis
            axis_shape = rot[..., 0:3].shape
            axis = torch.broadcast_to(axis, axis_shape)
            dof = dof.squeeze(-1)
            rot[:] = torch_util.axis_angle_to_quat(axis, dof)
        elif (self.joint_type == JointType.SPHERICAL):
            rot[:] = torch_util.exp_map_to_quat(dof)
        elif (self.joint_type == JointType.FIXED):
            rot[..., -1] = 1
        else:
            assert(False), "Unsupported joint type: {:s}".format(self.joint_type)

        return rot

    def rot_to_dof(self, rot):
        # Validate input is a tensor
        if not isinstance(rot, torch.Tensor):
            raise TypeError(
                f"Joint.rot_to_dof() expected torch.Tensor, got {type(rot).__name__}. "
                f"Joint: {self.name}, type: {self.joint_type}"
            )

        dof_dim = self.get_dof_dim()
        dof_shape = list(rot.shape[:-1])
        dof_shape = dof_shape + [dof_dim]
        dof = torch.zeros(dof_shape, device=rot.device, dtype=rot.dtype)

        if (self.joint_type == JointType.ROOT):
            pass
        elif (self.joint_type == JointType.HINGE):
            j_axis = self.axis
            angle = torch_util.quat_twist_angle(rot, j_axis)
            dof[:] = angle.unsqueeze(-1)
        elif (self.joint_type == JointType.SPHERICAL):
            dof[:] = torch_util.quat_to_exp_map(rot)
        elif (self.joint_type == JointType.FIXED):
            pass
        else:
            assert(False), "Unsupported joint type: {:s}".format(self.joint_type)

        return dof


class KinCharModel():
    def __init__(self, device):
        self._device = device
        return

    def init(self, body_names, parent_indices, local_translation, local_rotation, joints):
        num_bodies = len(body_names)
        assert(len(parent_indices) == num_bodies)
        assert(len(local_translation) == num_bodies)
        assert(len(local_rotation) == num_bodies)
        assert(len(joints) == num_bodies)

        # Validate all joints are Joint instances
        for i, joint in enumerate(joints):
            if joint is not None and not isinstance(joint, Joint):
                raise TypeError(
                    f"joints[{i}] expected Joint or None, got {type(joint).__name__}. "
                    f"Body name: {body_names[i]}"
                )

        self._body_names = body_names
        self._parent_indices = torch.tensor(parent_indices, device=self._device, dtype=torch.long)
        self._local_translation = torch.tensor(np.array(local_translation), device=self._device, dtype=torch.float32)
        self._local_rotation = torch.tensor(np.array(local_rotation), device=self._device, dtype=torch.float32)
        self._joints = joints

        self._dof_size = self._label_dof_indices(self._joints)
        self._name_body_map = self._build_name_body_map()
        return

    @abc.abstractmethod
    def load(self, char_file):
        return
    
    @abc.abstractmethod
    def save(self, output_file):
        return
    
    def get_body_names(self):
        return self._body_names

    def get_joint(self, j):
        assert(j > 0)
        return self._joints[j]

    def get_parent_id(self, j):
        return self._parent_indices[j]

    def get_dof_size(self):
        return self._dof_size

    def get_joint_dof_idx(self, j):
        dof_idx = self.get_joint(j).dof_idx
        return dof_idx

    def get_joint_dof_dim(self, j):
        dof_dim = self.get_joint(j).get_dof_dim()
        return dof_dim

    def get_num_joints(self):
        return len(self._joints)

    def dof_to_rot(self, dof):
        # Validate input is a tensor
        if not isinstance(dof, torch.Tensor):
            raise TypeError(
                f"KinCharModel.dof_to_rot() expected torch.Tensor, got {type(dof).__name__}. "
                f"dof value: {dof}"
            )

        num_joints = self.get_num_joints()

        # Validate we have joints initialized
        if num_joints == 0:
            raise ValueError("KinCharModel has no joints. Ensure init() was called.")

        rot_shape = list(dof.shape[:-1])
        rot_shape = rot_shape + [num_joints - 1, 4]
        joint_rot = torch.zeros(rot_shape, device=dof.device, dtype=dof.dtype)

        for j in range(1, num_joints):
            joint = self.get_joint(j)

            # Validate joint is properly initialized
            if not isinstance(joint, Joint):
                raise TypeError(
                    f"Expected Joint object at index {j}, got {type(joint).__name__}. "
                    f"Check _joints initialization in init()."
                )

            j_dof = joint.get_joint_dof(dof)
            j_rot = joint.dof_to_rot(j_dof)
            joint_rot[..., j - 1, :] = j_rot

        return joint_rot

    def rot_to_dof(self, rot):
        # Validate input is a tensor
        if not isinstance(rot, torch.Tensor):
            raise TypeError(
                f"KinCharModel.rot_to_dof() expected torch.Tensor, got {type(rot).__name__}"
            )

        dof_shape = list(rot.shape[:-2])
        dof_shape = dof_shape + [self._dof_size]
        dof = torch.zeros(dof_shape, device=rot.device, dtype=rot.dtype)

        num_joints = self.get_num_joints()
        for j in range(1, num_joints):
            joint = self.get_joint(j)
            j_dof_dim = joint.get_dof_dim()
            if (j_dof_dim > 0):
                j_rot = rot[..., j - 1, :]
                j_dof = joint.rot_to_dof(j_rot)
                joint.set_joint_dof(j_dof, dof)

        return dof

    def forward_kinematics(self, root_pos, root_rot, joint_rot):
        num_joints = self.get_num_joints()
        body_pos = [None] * num_joints
        body_rot = [None] * num_joints

        body_pos[0] = root_pos
        body_rot[0] = root_rot

        for j in range(1, num_joints):
            j_rot = joint_rot[..., j - 1, :]
            local_trans = self._local_translation[j]
            local_rot = self._local_rotation[j]
            parent_idx = self._parent_indices[j]
            
            parent_pos = body_pos[parent_idx]
            parent_rot = body_rot[parent_idx]

            local_trans = torch.broadcast_to(local_trans, parent_pos.shape)
            local_rot = torch.broadcast_to(local_rot, parent_rot.shape)

            world_trans = torch_util.quat_rotate(parent_rot, local_trans)

            curr_pos = parent_pos + world_trans
            curr_rot = torch_util.quat_mul(local_rot, j_rot)
            curr_rot = torch_util.quat_mul(parent_rot, curr_rot)

            body_pos[j] = curr_pos
            body_rot[j] = curr_rot

        body_pos = torch.stack(body_pos, dim=-2)
        body_rot = torch.stack(body_rot, dim=-2)
        
        return body_pos, body_rot
    
    def compute_frame_dof_vel(self, joint_rot, dt):
        joint_rot0 = joint_rot[:-1, :, :]
        joint_rot1 = joint_rot[1:, :, :]
        dof_vel = self.compute_dof_vel(joint_rot0, joint_rot1, dt)
        
        final_vels = dof_vel[-1:, :]
        dof_vel = torch.cat([dof_vel, final_vels], dim=-2)

        return dof_vel

    def compute_dof_vel(self, joint_rot0, joint_rot1, dt):
        dof_size = self.get_dof_size()
        dof_shape = list(joint_rot0.shape[:-2])
        dof_shape = dof_shape + [dof_size]
        dof_vel = torch.zeros(dof_shape, device=joint_rot0.device, dtype=joint_rot0.dtype)

        drot = torch_util.quat_mul(torch_util.quat_conjugate(joint_rot0), joint_rot1)
        drot = torch_util.quat_normalize(drot)

        num_joints = self.get_num_joints()
        for j in range(1, num_joints):
            joint = self.get_joint(j)
            j_drot = drot[..., j - 1, :]

            if (joint.joint_type == JointType.ROOT):
                pass
            elif (joint.joint_type == JointType.HINGE):
                j_axis = joint.axis
                j_dof_vel = torch_util.quat_to_exp_map(j_drot) / dt
                j_dof_vel = torch.sum(j_axis * j_dof_vel, dim=-1, keepdim=True)
                joint.set_joint_dof(j_dof_vel, dof_vel)
            elif (joint.joint_type == JointType.SPHERICAL):
                j_dof_vel = torch_util.quat_to_exp_map(j_drot) / dt
                joint.set_joint_dof(j_dof_vel, dof_vel)
            elif (joint.joint_type == JointType.FIXED):
                pass
            else:
                assert(False), "Unsupported joint type: {:s}".format(joint.joint_type)

        return dof_vel
    
    def get_body_name(self, body_id):
        return self._body_names[body_id]
    
    def get_body_id(self, body_name):
        assert body_name in self._name_body_map
        return self._name_body_map[body_name]
    
    def get_joint_id(self, body_name):
        assert body_name in self._name_body_map
        return self._name_body_map[body_name] -1 # joint arrays exclude the root
    
    def _build_name_body_map(self):
        name_body_map = dict()

        for body_id, body_name in enumerate(self._body_names):
            name_body_map[body_name] = body_id

        return name_body_map
    
    def _build_root_joint(self):
        joint = Joint(name="root",
                      joint_type=JointType.ROOT,
                      axis=None)
        return joint
    
    def _label_dof_indices(self, joints):
        dof_idx = 0
        for curr_joint in joints:
            if (curr_joint is not None):
                dof_dim = curr_joint.get_dof_dim()
                curr_joint.dof_idx = dof_idx
                dof_idx += dof_dim
        
        return dof_idx

    def _build_name_body_map(self):
        name_body_map = dict()

        for body_id, body_name in enumerate(self._body_names):
            name_body_map[body_name] = body_id

        return name_body_map
    
    def _build_body_children_map(self):
        num_joints = self.get_num_joints()
        body_children = [[] for _ in range(num_joints)]
        for j in range(num_joints):
            parent_idx = self._parent_indices[j].item()
            if (parent_idx != -1):
                body_children[parent_idx].append(j)
        
        return body_children