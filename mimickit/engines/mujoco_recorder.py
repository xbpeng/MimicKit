from __future__ import annotations

import mujoco
import numpy as np

import engines.video_recorder as video_recorder
from util import display
from util.logger import Logger


class MujocoVideoRecorder(video_recorder.VideoRecorder):
    def __init__(self,
                 engine,
                 resolution: tuple[int, int] = (854, 480),
                 cam_pos: np.array = np.array([-3.5, -3.5, 2.0]),
                 cam_target: np.array = np.array([0.0, 0.0, 1.0])) -> None:
        self._engine = engine
        self._env_id = 0
        self._obj_id = 0
        self._mj_model = engine.get_mj_model()
        self._mj_data = engine.create_mj_data()

        timestep = self._engine.get_timestep()
        fps = int(np.round(1.0 / timestep))
        super().__init__(fps, resolution, cam_pos, cam_target)

        display.ensure_virtual_display()
        self._renderer = mujoco.Renderer(
            model=self._mj_model,
            height=self._resolution[1],
            width=self._resolution[0],
        )
        self._camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self._mj_model, self._camera)
        Logger.print("[MujocoVideoRecorder] Created dedicated recording renderer at {:d}x{:d}".format(
            self._resolution[0], self._resolution[1]))
        return

    def _set_camera_pose(self):
        tar_pos = self._engine.get_root_pos(self._obj_id)[self._env_id].detach().cpu().numpy()
        tar_pos[2] = self._cam_target[2]
        cam_pos = tar_pos + (self._cam_pos - self._cam_target)

        cam_vec = cam_pos - tar_pos
        distance = max(float(np.linalg.norm(cam_vec)), 1e-5)

        self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE.value
        self._camera.fixedcamid = -1
        self._camera.trackbodyid = -1
        self._camera.lookat[:] = tar_pos
        self._camera.distance = distance
        self._camera.azimuth = np.rad2deg(np.arctan2(cam_vec[1], cam_vec[0]))
        self._camera.elevation = np.rad2deg(np.arcsin(np.clip(cam_vec[2] / distance, -1.0, 1.0)))
        return

    def _record_frame(self):
        self._set_camera_pose()
        self._engine.copy_env_state_to_mjdata(self._mj_data, self._env_id)
        mujoco.mj_forward(self._mj_model, self._mj_data)
        self._renderer.update_scene(self._mj_data, camera=self._camera)
        return self._renderer.render()
