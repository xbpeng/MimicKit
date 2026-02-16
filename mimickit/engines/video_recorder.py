import abc

import util.video as video

class VideoRecorder:
    def __init__(self, fps, resolution, cam_pos, cam_target):
        self._resolution = resolution
        self._cam_pos = cam_pos
        self._cam_target = cam_target
        self._video = video.Video(fps)
        return
    
    def clear(self):
        self._video.clear()
        return

    def capture_frame(self):
        frame = self._record_frame()
        assert(frame.shape[0] == self._resolution[1] and frame.shape[1] == self._resolution[0])

        self._video.add_frame(frame)
        return
    
    def get_video(self):
        return self._video
    
    def save(self, file_path):
        self._video.save(file_path)
        return
    
    @abc.abstractmethod
    def _record_Frame(self):
        return