from __future__ import annotations

import isaacgym.gymapi as gymapi
import numpy as np

import engines.video_recorder as video_recorder
import engines.isaac_gym_engine as isaac_gym_engine


class IsaacGymVideoRecorder(video_recorder.VideoRecorder):
    def __init__(self, 
                 engine: isaac_gym_engine.IsaacGymEngine, 
                 resolution: tuple[int, int] = (854, 480), 
                 cam_pos: np.array = np.array([-3.5, -3.5, 2.0]),
                 cam_target: np.array = np.array([0.0, 0.0, 1.0])):
        
        self._engine: isaac_gym_engine.IsaacGymEngine = engine
        self._env_id = 0
        self._obj_id = 0
        self._camera_ptr = None

        timestep = self._engine.get_timestep()
        fps = int(np.round(1.0 / timestep))
        super().__init__(fps, resolution, cam_pos, cam_target)
        return
    
    def _build_camera(self):
        camera_props = gymapi.CameraProperties()
        camera_props.width = self._resolution[0]
        camera_props.height = self._resolution[1]
        camera_props.horizontal_fov = 60.0

        env_ptr = self._engine.get_env(self._env_id)
        gym = self._engine.get_gym()
        camera_ptr = gym.create_camera_sensor(env_ptr, camera_props)
        assert(camera_ptr != -1), "Unable to create video camera."
        
        self._camera_ptr = camera_ptr
        return
    
    def _record_frame(self):
        if (self._camera_ptr is None):
            self._build_camera()

        tar_pos = self._engine.get_root_pos(self._obj_id)[self._env_id].cpu().numpy()
        tar_pos[2] = self._cam_target[2]
        cam_delta = self._cam_pos - self._cam_target
        cam_pos = tar_pos + cam_delta
        self._set_camera_pose(cam_pos, tar_pos)
        
        gym = self._engine.get_gym()
        sim = self._engine.get_sim()
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        gym.start_access_image_tensors(sim)

        env_ptr = self._engine.get_env(self._env_id)

        rgb_data = gym.get_camera_image(sim, env_ptr, self._camera_ptr, gymapi.IMAGE_COLOR)
        assert(rgb_data is not None), "Failed to render image."

        frame = np.frombuffer(rgb_data, dtype=np.uint8).reshape(self._resolution[1], self._resolution[0], 4)
        frame = frame[:, :, :3]  # drop alpha channel
        return frame
    
    def _set_camera_pose(self, cam_pos, cam_target):
        gym_cam_pos = gymapi.Vec3(cam_pos[0], cam_pos[1], cam_pos[2])
        gym_cam_target = gymapi.Vec3(cam_target[0], cam_target[1], cam_target[2])

        gym = self._engine.get_gym()
        env_ptr = self._engine.get_env(self._env_id)
        gym.set_camera_location(self._camera_ptr, env_ptr, gym_cam_pos, gym_cam_target)
        return