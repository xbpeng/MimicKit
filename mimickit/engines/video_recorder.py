import abc

import util.video as video

class VideoRecorder:
    def __init__(self, fps, resolution):
        self._resolution= resolution
        self._video = video.Video(fps)
        return
    
    def clear(self):
        self._video.clear()
        return

    def capture_frame(self):
        frame = self._record_frame()
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