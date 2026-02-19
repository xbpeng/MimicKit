from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Any

import engines.video_recorder as video_recorder
from util import display
from util.logger import Logger

if TYPE_CHECKING:
    import engines.isaac_lab_engine as isaac_lab_engine


class IsaacLabVideoRecorder(video_recorder.VideoRecorder):
    def __init__(self, 
                 engine: isaac_lab_engine.IsaacLabEngine,
                 resolution: tuple[int, int] = (854, 480), 
                 cam_pos: np.array = np.array([-3.5, -3.5, 2.0]),
                 cam_target: np.array = np.array([0.0, 0.0, 1.0])) -> None:
        
        self._engine: isaac_lab_engine.IsaacLabEngine = engine
        self._env_id = 0
        self._obj_id = 0
        self._annotator = None
        self._render_product = None

        timestep = self._engine.get_timestep()
        fps = int(np.round(1.0 / timestep))
        super().__init__(fps, resolution, cam_pos, cam_target)
        
        display.ensure_virtual_display()
        return
    
    def _record_frame(self):
        self._build_annotator()

        tar_pos = self._engine.get_root_pos(self._obj_id)[self._env_id].cpu().numpy()
        tar_pos[2] = self._cam_target[2]
        cam_delta = self._cam_pos - self._cam_target
        cam_pos = tar_pos + cam_delta
        self._engine.set_camera_pose(cam_pos, tar_pos)

        sim = self._engine.get_sim()
        sim.render()

        rgb_data: Any = self._annotator.get_data()
        if rgb_data is None or rgb_data.size == 0:
            # Renderer still warming up
            frame: np.ndarray = np.zeros((self._resolution[1], self._resolution[0], 3), dtype=np.uint8)
        else:
            frame = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            frame = frame[:, :, :3]  # drop alpha channel

        return frame
    
    def _build_annotator(self):
        import omni.replicator.core as rep

        video_cam_path = "/OmniverseKit_Persp"  # Use viewport camera (works in headless mode)

        """Lazily create the render product and RGB annotator."""
        if self._annotator is None:
            self._render_product = rep.create.render_product(video_cam_path, self._resolution)

            self._annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            self._annotator.attach([self._render_product])
            Logger.print("[VideoRecorder] Created RGB annotator for {}".format(video_cam_path))
        return
