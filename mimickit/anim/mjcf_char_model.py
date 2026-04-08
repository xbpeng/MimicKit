import numpy as np
import torch
import xml.etree.ElementTree as ET

import anim.kin_char_model as kin_char_model
import util.torch_util as torch_util

#######################################
## MJCF Character Model
#######################################

class MJCFCharModel(kin_char_model.KinCharModel):
    def __init__(self, device):
        super().__init__(device)
        return
    
    def load(self, char_file):
        tree = ET.parse(char_file)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")
        assert(xml_world_body is not None)

        xml_body_root = xml_world_body.find("body")
        assert(xml_body_root is not None)

        body_names = []
        parent_indices = []
        local_translation = []
        local_rotation = []
        joints = []

        default_joint_type = self._parse_default_joint_type(xml_doc_root)

        # recursively adding all bodies into the skel_tree
        def _add_xml_body(xml_node, parent_index, body_index, default_joint_type):
            body_name = xml_node.attrib.get("name")
            # parse the local translation into float list
            pos_data = xml_node.attrib.get("pos")
            if (pos_data is None):
                pos = np.array([0.0, 0.0, 0.0])
            else:
                pos = np.fromstring(pos_data, dtype=float, sep=" ")
            
            rot_data = xml_node.attrib.get("quat")
            if (rot_data is None):
                rot = np.array([0.0, 0.0, 0.0, 1.0])
            else:
                rot = np.fromstring(rot_data, dtype=float, sep=" ")
                rot_w = rot[..., 0].copy()
                rot[..., 0:3] = rot[..., 1:]
                rot[..., -1] = rot_w
                rot = rot / np.linalg.norm(rot)

            if (body_index == 0):
                curr_joint = self._build_root_joint()
            else:
                joint_data = xml_node.findall("joint")
                curr_joint = self._parse_joint(body_name, joint_data, default_joint_type)

            body_names.append(body_name)
            parent_indices.append(parent_index)
            local_translation.append(pos)
            local_rotation.append(rot)
            joints.append(curr_joint)

            curr_index = body_index
            body_index += 1
            for child_body in xml_node.findall("body"):
                body_index = _add_xml_body(child_body, curr_index, body_index, default_joint_type)

            return body_index

        _add_xml_body(xml_body_root, -1, 0, default_joint_type)

        self.init(body_names=body_names,
                  parent_indices=parent_indices,
                  local_translation=local_translation,
                  local_rotation=local_rotation,
                  joints=joints)
        return
     
    def save(self, output_file):
        xml_template = """<mujoco model="character">
    <compiler angle="radian"/>
  
    <default>
        <motor ctrlrange="-1 1" ctrllimited="true"/>
        <default class="body">
            <geom condim="1" friction="1.0 0.05 0.05" solimp=".9 .99 .003" solref=".015 1"/>
            <joint limited="true" solimplimit="0 .99 .01"/>
        </default>
    </default>

    <worldbody>
{:s}
    </worldbody>

    <actuator>{:s}
    </actuator>
</mujoco>"""

        bodies_xml = self._build_bodies_xml()
        actuator_xml = self._build_actuators_xml()

        char_xml = xml_template.format(bodies_xml, actuator_xml)

        with open(output_file, "w") as out_file:
            out_file.write(char_xml)

        return

    def _parse_joint(self, body_name, xml_joint_data, default_joint_type):
        num_joints = len(xml_joint_data)

        if (num_joints == 0):
            joint = self._parse_fixed_joint(body_name)
        elif (num_joints == 3):
            joint = self._parse_sphere_joint(xml_joint_data, default_joint_type)
        elif (num_joints == 1):
            joint_type_str = xml_joint_data[0].attrib.get("type")
            if (joint_type_str is None):
                joint_type_str = default_joint_type

            if (joint_type_str == "hinge"):
                joint = self._parse_hinge_joint(xml_joint_data[0])
            else:
                assert(False), "Unsupported joint type: {:s}".format(joint_type_str)
        else:
            assert(False), "Series joints are not supported."
        
        return joint

    def _parse_hinge_joint(self, xml_joint_data):
        joint_name = xml_joint_data.attrib.get("name")

        joint_pos_data = xml_joint_data.attrib.get("pos")
        if (joint_pos_data is not None):
            joint_pos = np.fromstring(joint_pos_data, dtype=float, sep=" ")
            if (np.any(joint_pos)):
                assert(False), "Joint offsets are not supported"

        joint_axis = np.fromstring(xml_joint_data.attrib.get("axis"), dtype=float, sep=" ")
        joint_axis = torch.tensor(joint_axis, device=self._device, dtype=torch.float32)

        joint = kin_char_model.Joint(name=joint_name,
                      joint_type=kin_char_model.JointType.HINGE,
                      axis=joint_axis)
        return joint

    def _parse_sphere_joint(self, xml_joint_data, default_joint_type):
        # consolidate series of three hinge joints into a single spherical joint
        num_joints = len(xml_joint_data)
        assert(num_joints == 3)

        is_spherical = True
        for joint_data in xml_joint_data:
            joint_type_str = joint_data.attrib.get("type")
            if (joint_type_str is None):
                joint_type_str = default_joint_type

            joint_pos_data = joint_data.attrib.get("pos")
            if (joint_pos_data is not None):
                joint_pos = np.fromstring(joint_pos_data, dtype=float, sep=" ")
                if (np.any(joint_pos)):
                    assert(False), "Joint offsets are not supported"

            if (joint_type_str != "hinge"):
                is_spherical = False
                break

        if (is_spherical):
            joint_name = xml_joint_data[0].attrib.get("name")
            joint_name = joint_name[:joint_name.rfind('_')]
            joint = kin_char_model.Joint(name=joint_name,
                               joint_type=kin_char_model.JointType.SPHERICAL,
                               axis=None)
        else:
            assert(False), "Invalid format for a spherical joint"

        return joint

    def _parse_fixed_joint(self, body_name):
        joint = kin_char_model.Joint(name=body_name,
                      joint_type=kin_char_model.JointType.FIXED,
                      axis=None)

        return joint
    
    def _parse_default_joint_type(self, xml_node):
        joint_type_str = "hinge"

        default_data = xml_node.find("default")

        if (default_data is not None):
            default_data = default_data.findall("default")

            if (default_data is not None):
                for data in default_data:
                    class_data = data.attrib.get("class")
                    if (class_data == "body"):
                        joint_data = data.find("joint")
                        if (joint_data is not None):
                            joint_type_str = joint_data.attrib.get("type")
                            break

        return joint_type_str
    
    def _build_bodies_xml(self):
        body_children = self._build_body_children_map(self._parent_indices)
        bodies_xml = self._build_body_xml(body_children=body_children, body_id=0)
        return bodies_xml

    def _build_body_xml(self, body_children, body_id):
        body_template = '''
        <body name="{:s}" pos="{:.4f} {:.4f} {:.4f}" rot="{:.4f} {:.4f} {:.4f} {:.4f}">{:s}{:s}{:s}
        </body>'''

        root_template = '''
        <body name="{:s}" pos="0 0 0" childclass="body">{:s}{:s}{:s}
        </body>'''
        
        body_name = self._body_names[body_id]
        pos = self._local_translation[body_id].cpu().numpy()
        rot = self._local_rotation[body_id].cpu().numpy()

        children_xml = ""
        children = body_children[body_id]
        if (len(children) > 0):
            for child_id in children:
                child_xml = self._build_body_xml(body_children=body_children, body_id=child_id)

                child_xml_lines = child_xml.splitlines()
                indented_xml_lines = ["\t" + l for l in child_xml_lines]
                child_xml = "\n".join(indented_xml_lines)

                children_xml += "\n" + child_xml

        joint_xml = self._build_joint_xml(body_id)
        geom_xml = self._build_geom_xml(body_children, body_id)

        is_root = body_id == 0
        if (is_root):
            body_xml = root_template.format(body_name,
                                            joint_xml,
                                            geom_xml,
                                            children_xml)
        else:
            body_xml = body_template.format(body_name,
                                            pos[0], pos[1], pos[2], 
                                            rot[3], rot[0], rot[1], rot[2], 
                                            joint_xml,
                                            geom_xml,
                                            children_xml)
        return body_xml

    def _build_joint_xml(self, body_id):
        root_template = '''
			<freejoint name="{:s}"/>'''

        joint_template = '''
            <joint name="{:s}_x" type="hinge" axis="1 0 0" range="-3.14 3.14" actuatorfrcrange="-100 100" stiffness="100" damping="10" armature=".01"/>
            <joint name="{:s}_y" type="hinge" axis="0 1 0" range="-3.14 3.14" actuatorfrcrange="-100 100" stiffness="100" damping="10" armature=".01"/>
            <joint name="{:s}_z" type="hinge" axis="0 0 1" range="-3.14 3.14" actuatorfrcrange="-100 100" stiffness="100" damping="10" armature=".01"/>'''
            
        body_name = self._body_names[body_id]
        
        is_root = body_id == 0
        if (is_root):
            joint_xml = root_template.format(body_name)
        else:
            joint = self.get_joint(body_id)
            joint_type = joint.joint_type

            if (joint_type == kin_char_model.JointType.HINGE):
                j_axis = joint.axis
                joint_template = '''
            <joint name="{:s}" type="hinge" axis="{:.4f} {:.4f} {:.4f}" range="-3.14 3.14" stiffness="100" damping="10" armature=".01"/>'''
                joint_xml = joint_template.format(body_name, j_axis[0], j_axis[1], j_axis[2])
            elif (joint_type == kin_char_model.JointType.SPHERICAL):
                joint_template = '''
            <joint name="{:s}_x" type="hinge" axis="1 0 0" range="-3.14 3.14" stiffness="100" damping="10" armature=".01"/>
            <joint name="{:s}_y" type="hinge" axis="0 1 0" range="-3.14 3.14" stiffness="100" damping="10" armature=".01"/>
            <joint name="{:s}_z" type="hinge" axis="0 0 1" range="-3.14 3.14" stiffness="100" damping="10" armature=".01"/>'''
                joint_xml = joint_template.format(body_name, body_name, body_name)
            elif (joint_type == kin_char_model.JointType.FIXED):
                joint_xml = ""
            else:
                assert(False), "Unsupported joint type: {:s}".format(joint.joint_type)

        return joint_xml

    def _build_geom_xml(self, body_children, body_id):
        joint_radius = 0.02
        bone_radius = 0.01
        joint_template = '''
            <geom type="sphere" name="{:s}" pos="0 0 0" size="{:.4f}" density="1000"/>'''
        bone_template = '''
            <geom type="capsule" name="{:s}" fromto="0 0 0 {:.4f} {:.4f} {:.4f}" size="{:.4f}" density="1000"/>'''

        body_name = self._body_names[body_id]

        geom_xml = ""

        joint_pos = self._local_translation[body_id]
        bone_len = np.linalg.norm(joint_pos)
        if (bone_len > 0):
            clamp_joint_radius = min(0.25 * bone_len, joint_radius)
            geom_xml += joint_template.format(body_name, clamp_joint_radius)

        children = body_children[body_id]
        joint_rot = self._local_rotation[body_id]
        for c in children:
            child_pos = self._local_translation[c]
            bone_pos = torch_util.quat_rotate(joint_rot, child_pos)

            child_bone_len = np.linalg.norm(bone_pos)
            if (child_bone_len > 0):
                geom_len = max(0.001, child_bone_len)
                geom_pos = geom_len * bone_pos / child_bone_len
                child_bone_radius = min(0.15 * child_bone_len, bone_radius)

                bone_xml = bone_template.format(body_name, geom_pos[0], geom_pos[1], geom_pos[2], child_bone_radius)
                geom_xml += bone_xml

        return geom_xml

    def _build_actuators_xml(self):
        motor_template = '''
        <motor name='{:s}'       	gear='100' 	joint='{:s}'/>'''

        actuators_xml = ""

        num_joints = self.get_num_joints()
        for j in range(1, num_joints):
            joint = self.get_joint(j)
            joint_type = joint.joint_type
            body_name = self._body_names[j]

            if (joint_type == kin_char_model.JointType.HINGE):
                actuator_xml = motor_template.format(body_name, body_name)
            elif (joint_type == kin_char_model.JointType.SPHERICAL):
                actuator_xml = motor_template.format(body_name + "_x", body_name + "_x")
                actuator_xml += motor_template.format(body_name + "_y", body_name + "_y")
                actuator_xml += motor_template.format(body_name + "_z", body_name + "_z")
            elif (joint_type == kin_char_model.JointType.FIXED):
                actuator_xml = ""
            else:
                assert(False), "Unsupported joint type: {:s}".format(joint.joint_type)

            actuators_xml += actuator_xml

        return actuators_xml