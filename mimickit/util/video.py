from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

class Video:
    def __init__(self, fps):
        self._fps = fps
        self._frames = []
        return
    
    def clear(self):
        self._frames = []
        return
    
    def get_fps(self):
        return self._fps
    
    def get_num_frames(self):
        return len(self._frames)
    
    def get_resolution(self):
        num_frames = self.get_num_frames()
        if (num_frames == 0):
            res = (0, 0)
        else:
            frame = self._frames[0]
            res = (frame.shape[0], frame.shape[1])
        return res

    def add_frame(self, frame):
        assert(len(frame.shape) == 3)

        num_frames = self.get_num_frames()
        if (num_frames > 0):
            res = self.get_resolution()
            assert(res[0] == frame.shape[0] and res[1] == frame.shape[1])

        self._frames.append(frame)
        return
    
    def get_frames(self):
        return self._frames
    
    def save(self, file_path):
        clip = ImageSequenceClip(self._frames, fps=self._fps)
        clip.write_videofile(file_path, logger=None)
        return
