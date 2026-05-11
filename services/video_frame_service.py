import os

import cv2


class VideoFrameService:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def extract_last_frame(self, video_path: str, clip_id: int) -> str:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise IOError(f"Cannot open video {video_path}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            capture.release()
            raise IOError("Cannot extract frame: video has no frames.")

        capture.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ok, frame = capture.read()
        capture.release()

        if not ok:
            raise IOError("Failed to read last frame.")

        filename = f"frame_end_clip_{clip_id:03d}.jpg"
        save_path = os.path.join(self.output_dir, filename)
        saved = cv2.imwrite(save_path, frame)
        if not saved:
            raise IOError(f"Failed to save frame to {save_path}")
        return save_path
