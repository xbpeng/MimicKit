import abc
import enum

class ControlMode(enum.Enum):
    none = 0
    pos = 1
    vel = 2
    torque = 3
    pd_explicit = 4

class ObjType(enum.Enum):
    rigid = 0
    articulated = 1

class Engine:
    def __init__(self):
        return
    
    @abc.abstractmethod
    def get_name(self):
        return
    
    @abc.abstractmethod
    def create_env(self, env_id):
        return
    
    @abc.abstractmethod
    def create_obj(self, env_id, obj_type, asset_file, name, is_visual=False, enable_self_collisions=True, 
                     fix_root=False, start_pos=None, start_rot=None, color=None, disable_motors=False):
        return

    @abc.abstractmethod
    def initialize_sim(self):
        return
    
    @abc.abstractmethod
    def set_cmd(self, obj_id, cmd):
        return

    @abc.abstractmethod
    def step(self):
        return
    
    @abc.abstractmethod
    def set_camera_pose(self, pos, look_at):
        return
    
    @abc.abstractmethod
    def get_camera_pos(self):
        return
    
    @abc.abstractmethod
    def get_camera_dir(self):
        return
    
    @abc.abstractmethod
    def render(self):
        return
    
    @abc.abstractmethod
    def get_timestep(self):
        return 0.0
    
    @abc.abstractmethod
    def get_num_envs(self):
        return 0
    
    @abc.abstractmethod
    def get_root_pos(self, obj_id):
        return
    
    @abc.abstractmethod
    def get_root_rot(self, obj_id):
        return
    
    @abc.abstractmethod
    def get_root_vel(self, obj_id):
        return
    
    @abc.abstractmethod
    def get_root_ang_vel(self, obj_id):
        return
    
    @abc.abstractmethod
    def get_dof_pos(self, obj_id):
        return
    
    @abc.abstractmethod
    def get_dof_vel(self, obj_id):
        return
    
    @abc.abstractmethod
    def get_dof_forces(self, obj_id):
        return
    
    @abc.abstractmethod
    def get_body_pos(self, obj_id):
        return
    
    @abc.abstractmethod
    def get_body_rot(self, obj_id):
        return
    
    @abc.abstractmethod
    def get_body_vel(self, obj_id):
        return
    
    @abc.abstractmethod
    def get_body_ang_vel(self, obj_id):
        return
    
    @abc.abstractmethod
    def get_contact_forces(self, obj_id):
        return
    
    @abc.abstractmethod
    def get_ground_contact_forces(self, obj_id):
        return
    
    @abc.abstractmethod
    def set_root_pos(self, env_id, obj_id, root_pos):
        return
    
    @abc.abstractmethod
    def set_root_rot(self, env_id, obj_id, root_rot):
        return
    
    @abc.abstractmethod
    def set_root_vel(self, env_id, obj_id, root_vel):
        return
    
    @abc.abstractmethod
    def set_root_ang_vel(self, env_id, obj_id, root_ang_vel):
        return
    
    @abc.abstractmethod
    def set_dof_pos(self, env_id, obj_id, dof_pos):
        return
    
    @abc.abstractmethod
    def set_dof_vel(self, env_id, obj_id, dof_vel):
        return
    
    @abc.abstractmethod
    def set_body_vel(self, env_id, obj_id, body_vel):
        return
    
    @abc.abstractmethod
    def set_body_ang_vel(self, env_id, obj_id, body_ang_vel):
        return
    
    @abc.abstractmethod
    def set_body_forces(self, env_id, obj_id, body_id, forces):
        return
    
    @abc.abstractmethod
    def get_obj_type(self, obj_id):
        return
    
    @abc.abstractmethod
    def get_obj_num_dofs(self, obj_id):
        return
    
    @abc.abstractmethod
    def get_obj_num_bodies(self, obj_id):
        return
    
    @abc.abstractmethod
    def get_obj_body_names(self, obj_id):
        return

    @abc.abstractmethod
    def find_obj_body_id(self, obj_id, body_name):
        return 
    
    @abc.abstractmethod
    def get_obj_torque_limits(self, env_id, obj_id):
        return
    
    @abc.abstractmethod
    def get_obj_dof_limits(self, env_id, obj_id):
        return
    
    @abc.abstractmethod
    def calc_obj_mass(self, env_id, obj_id):
        return
    
    @abc.abstractmethod
    def get_control_mode(self):
        return

    @abc.abstractmethod
    def create_video_recorder(self):
        return
    
    def draw_lines(self, env_id, start_verts, end_verts, cols, line_width):
        return
    
    def register_keyboard_callback(self, key_str, callback_func):
        return
    
    def enabled_record_video(self):
        return False
    
    def get_video_recording(self):
        return None
    
    def start_video_recording(self):
        return
    
    def stop_video_recording(self):
        return