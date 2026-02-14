from __future__ import annotations

import numpy as np
import os
import subprocess
import time
from typing import TYPE_CHECKING, Any

import engines.video_recorder as video_recorder
from util.logger import Logger

if TYPE_CHECKING:
    import engines.isaac_lab_engine as isaac_lab_engine


class IsaacLabVideoRecorder(video_recorder.VideoRecorder):
    """Records video frames from the simulation and uploads to WandB.
    
    Works with Isaac Lab engine in headless mode using the omni.replicator
    annotator API to capture viewport images.

    The recorder manages its own camera controls, independent of the environment's
    visualization camera. This allows recording without interfering with visualization.
    
    Args:
        engine: The simulation engine (e.g. IsaacLabEngine).
        cam_pos: np.ndarray, camera position [x, y, z]. Defaults to [-4, -4, 2].
        cam_target: np.ndarray, camera target [x, y, z]. Defaults to [0, 0, 1].
        resolution: Tuple (width, height) for the captured frames.
        fps: Frames per second for the output video.
    """

    def __init__(self, 
                 engine: isaac_lab_engine.IsaacLabEngine, 
                 fps: int, 
                 cam_pos: np.array = np.array([-4.0, -4.0, 2.0]),
                 cam_target: np.array = np.array([0.0, 0.0, 1.0]),
                 resolution: tuple[int, int] = (854, 480)) -> None:
        
        super().__init__(fps, resolution)

        self._engine: isaac_lab_engine.IsaacLabEngine = engine
        self._cam_pos = cam_pos
        self._cam_target = cam_target
        self._annotator = None
        self._render_product = None
        
        self._ensure_virtual_display()
        return
    
    def _record_frame(self):
        self._ensure_annotator()
        self._engine.set_camera_pose(self._cam_pos, self._cam_target)
            
        # Render the scene to update the viewport
        self._engine.render()

        rgb_data: Any = self._annotator.get_data()
        if rgb_data is None or rgb_data.size == 0:
            # Renderer still warming up
            frame: np.ndarray = np.zeros((self._resolution[1], self._resolution[0], 3), dtype=np.uint8)
        else:
            frame = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            frame = frame[:, :, :3]  # drop alpha channel

        return frame
    
    def _ensure_annotator(self):
        import omni.replicator.core as rep

        video_cam_path = "/OmniverseKit_Persp"  # Use viewport camera (works in headless mode)

        """Lazily create the render product and RGB annotator."""
        if self._annotator is None:
            self._render_product = rep.create.render_product(video_cam_path, self._resolution)

            self._annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            self._annotator.attach([self._render_product])
            Logger.print("[VideoRecorder] Created RGB annotator for {}".format(video_cam_path))
        return

    def _ensure_virtual_display(self, display=":99") -> None:
        """Start Xvfb virtual display if no DISPLAY is set. Needed for headless Vulkan rendering.
        
        If DISPLAY is already set, uses it (assumes it's valid). Otherwise starts Xvfb on the
        specified display number.
        """
        if "DISPLAY" not in os.environ:
            try:
                process: subprocess.Popen[bytes] = subprocess.Popen(
                    ["Xvfb", display, "-screen", "0", "1024x768x24"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                time.sleep(1)
                os.environ["DISPLAY"] = display
                Logger.print("Started virtual display on {}".format(display))
            except FileNotFoundError:
                Logger.print("WARNING: Xvfb not found. Install with: apt-get install xvfb")
                Logger.print("Headless camera rendering may not work without a virtual display.")
        return
