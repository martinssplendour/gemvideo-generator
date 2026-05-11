import os
import time
from dataclasses import dataclass
from typing import List, Optional

from google.genai import types

from services.storage_service import StorageService
from services.veo_client_service import VeoClientService
from services.video_frame_service import VideoFrameService


@dataclass(frozen=True)
class ClipGenerationResult:
    clip_filename: str
    end_frame_filename: str
    clip_id: int
    active_reference_count: int


class VeoDirectorService:
    def __init__(
        self,
        storage_service: StorageService,
        frame_service: VideoFrameService,
        veo_client_service: VeoClientService,
        max_retries: int = 3,
        retry_delay_seconds: int = 5,
    ):
        self.storage_service = storage_service
        self.frame_service = frame_service
        self.veo_client_service = veo_client_service
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

        self.active_ref_paths: List[str] = []
        self.last_frame_path: Optional[str] = None
        self.clip_count = 0

    def add_references(self, file_paths: List[str]) -> None:
        self.active_ref_paths.extend(file_paths)
        print(f"Session now has {len(self.active_ref_paths)} active reference images.")

    def generate_clip(self, prompt: str) -> ClipGenerationResult:
        self.clip_count += 1
        full_prompt = self._build_prompt(prompt, is_continuation=(self.clip_count > 1))
        reference_images = self._build_reference_images()

        print(
            f"Generating Clip {self.clip_count} with {len(reference_images)} references..."
        )
        print(f"PROMPT SENT: {full_prompt}")

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if attempt > 1:
                    print(f"Retry attempt {attempt}/{self.max_retries}...")
                    time.sleep(self.retry_delay_seconds)

                video_bytes = self.veo_client_service.generate_video_bytes(
                    prompt=full_prompt,
                    reference_images=reference_images,
                )
                video_path = self.storage_service.save_clip_video(
                    clip_id=self.clip_count,
                    video_bytes=video_bytes,
                )
                print(f"Video saved: {video_path}")

                end_frame_path = self.frame_service.extract_last_frame(
                    video_path=video_path,
                    clip_id=self.clip_count,
                )
                self.last_frame_path = end_frame_path

                return ClipGenerationResult(
                    clip_filename=os.path.basename(video_path),
                    end_frame_filename=os.path.basename(end_frame_path),
                    clip_id=self.clip_count,
                    active_reference_count=len(self.active_ref_paths),
                )
            except Exception as error:
                last_error = error
                print(f"\nException on attempt {attempt}: {error}")

        raise RuntimeError(
            f"Failed after {self.max_retries} attempts. Last error: {last_error}"
        )

    @staticmethod
    def _build_prompt(user_prompt: str, is_continuation: bool) -> str:
        if not is_continuation:
            return user_prompt
        return (
            "CONTINUITY MODE: Start exactly where the previous shot ended. "
            "Maintain exact character positioning, lighting, environment, and camera angle. "
            "Smooth temporal transition. "
            f"\n\n{user_prompt}"
        )

    def _build_reference_images(self) -> List[types.VideoGenerationReferenceImage]:
        current_refs: List[types.VideoGenerationReferenceImage] = []

        if self.last_frame_path:
            print("Using previous clip's end frame for continuity.")
            current_refs.append(self._read_reference(self.last_frame_path))

        for path in self.active_ref_paths:
            current_refs.append(self._read_reference(path))

        return current_refs

    def _read_reference(self, path: str) -> types.VideoGenerationReferenceImage:
        return types.VideoGenerationReferenceImage(
            image=types.Image(image_bytes=self.storage_service.read_bytes(path))
        )
