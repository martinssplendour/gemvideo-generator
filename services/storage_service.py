import os
import uuid
from typing import Iterable, List

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename


class StorageService:
    def __init__(self, output_dir: str, upload_dir: str):
        self.output_dir = output_dir
        self.upload_dir = upload_dir

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.upload_dir, exist_ok=True)

    def save_uploads(self, files: Iterable[FileStorage]) -> List[str]:
        saved_paths: List[str] = []
        for file in files:
            if not file or not file.filename:
                continue

            filename = secure_filename(file.filename)
            if not filename:
                continue

            save_path = os.path.join(self.upload_dir, filename)
            file.save(save_path)
            saved_paths.append(save_path)
        return saved_paths

    @staticmethod
    def read_bytes(file_path: str) -> bytes:
        with open(file_path, "rb") as file:
            return file.read()

    def save_clip_video(self, clip_id: int, video_bytes: bytes) -> str:
        filename = f"clip_{clip_id:03d}.mp4"
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "wb") as file:
            file.write(video_bytes)
        return output_path

    def save_generated_video(self, video_bytes: bytes, prefix: str = "video") -> str:
        safe_prefix = secure_filename(prefix) or "video"
        filename = f"{safe_prefix}_{uuid.uuid4().hex[:10]}.mp4"
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "wb") as file:
            file.write(video_bytes)
        return output_path
