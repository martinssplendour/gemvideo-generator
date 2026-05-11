from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass
from typing import Any, Optional

from google.genai import types
from werkzeug.datastructures import FileStorage

from services.storage_service import StorageService


@dataclass(frozen=True)
class SimpleVideoResult:
    output_filename: str
    output_path: str
    provider: str
    model_name: str
    used_reference_image: bool
    aspect_ratio: str


class SimpleVideoService:
    def __init__(
        self,
        storage_service: StorageService,
        video_client_service: Any,
        provider: str,
        model_name: str,
    ):
        self.storage_service = storage_service
        self.video_client_service = video_client_service
        self.provider = str(provider or "").strip() or "unknown"
        self.model_name = str(model_name or "").strip() or "unknown"

    def generate_video(
        self,
        prompt: str,
        image_upload: Optional[FileStorage] = None,
    ) -> SimpleVideoResult:
        cleaned_prompt = str(prompt or "").strip()
        if not cleaned_prompt:
            raise ValueError("Prompt is required.")

        reference_images = []
        used_reference_image = False
        aspect_ratio = ""

        if image_upload and image_upload.filename:
            reference_images.append(self._build_reference_image(image_upload))
            used_reference_image = True
            # Veo reference-image flows are more reliable in 16:9.
            aspect_ratio = "16:9"

        video_bytes = self.video_client_service.generate_video_bytes(
            prompt=cleaned_prompt,
            reference_images=reference_images,
            aspect_ratio=aspect_ratio or None,
            last_frame=None,
            allow_reference_fallback=False,
        )
        output_path = self.storage_service.save_generated_video(
            video_bytes=video_bytes,
            prefix="single_video",
        )
        return SimpleVideoResult(
            output_filename=os.path.basename(output_path),
            output_path=output_path,
            provider=self.provider,
            model_name=self.model_name,
            used_reference_image=used_reference_image,
            aspect_ratio=aspect_ratio,
        )

    def _build_reference_image(
        self,
        image_upload: FileStorage,
    ) -> types.VideoGenerationReferenceImage:
        image_bytes = image_upload.read()
        if not image_bytes:
            raise ValueError("Uploaded image is empty.")
        mime_type = self._resolve_mime_type(image_upload)
        return types.VideoGenerationReferenceImage(
            image=types.Image(image_bytes=image_bytes, mime_type=mime_type)
        )

    @staticmethod
    def _resolve_mime_type(image_upload: FileStorage) -> str:
        mime_type = str(getattr(image_upload, "mimetype", "") or "").strip().lower()
        if mime_type.startswith("image/"):
            return mime_type
        guessed, _ = mimetypes.guess_type(str(image_upload.filename or ""))
        if guessed and guessed.startswith("image/"):
            return guessed
        return "image/png"
