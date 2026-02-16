from __future__ import annotations

import newton
import numpy as np
import pyglet
from typing import TYPE_CHECKING

import engines.video_recorder as video_recorder
from util.logger import Logger

if TYPE_CHECKING:
    import engines.newton_engine as newton_engine


class NewtonVideoRecorder(video_recorder.VideoRecorder):
    def __init__(self,
                 engine: newton_engine.NewtonEngine,
                 resolution: tuple[int, int] = (854, 480),
                 cam_pos: np.array = np.array([-3.5, -3.5, 2.0]),
                 cam_target: np.array = np.array([0.0, 0.0, 1.0])) -> None:

        self._engine: newton_engine.NewtonEngine = engine
        self._env_id = 0
        self._obj_id = 0

        timestep = self._engine.get_timestep()
        fps = int(np.round(1.0 / timestep))
        super().__init__(fps, resolution, cam_pos, cam_target)

        self._build_viewer()
        self._cam_offset = self._viewer.world_offsets.numpy()[self._env_id]
        return

    def _build_viewer(self):
        """Create a separate headless viewer specifically for video recording."""
        self._viewer = newton.viewer.ViewerGL(
            width=self._resolution[0],
            height=self._resolution[1],
            headless=True
        )
        self._engine.init_viewer(self._viewer)
        Logger.print(f"[NewtonVideoRecorder] Created dedicated recording viewer at {self._resolution[0]}x{self._resolution[1]}")
        return
    
    def _set_camera_pose(self):
        """Set camera position and orientation to track the target object."""
        tar_pos = self._engine.get_root_pos(self._obj_id)[self._env_id].cpu().numpy()
        tar_pos[2] = self._cam_target[2]
        cam_pos = tar_pos + (self._cam_pos - self._cam_target)
        
        direction = tar_pos - cam_pos
        dx, dy, dz = direction
        pitch = np.arctan2(dz, np.hypot(dx, dy))
        yaw = np.arctan2(dy, dx)

        cam_pos_with_offset = cam_pos + self._cam_offset
        self._viewer.set_camera(
            pyglet.math.Vec3(*cam_pos_with_offset),
            np.rad2deg(pitch),
            np.rad2deg(yaw)
        )
        return

    def _record_frame(self):
        self._set_camera_pose()
        sim_time = self._engine.get_sim_time()
        sim_state = self._engine.get_sim_state()

        self._viewer.begin_frame(sim_time)
        self._viewer.log_state(sim_state.raw_state)
        self._viewer.end_frame()

        try:
            frame = self._viewer.get_frame(render_ui=False).numpy()
        except Exception as e:
            frame = np.zeros((self._resolution[1], self._resolution[0], 3), dtype=np.uint8)
        return frame
