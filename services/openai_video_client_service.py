import base64
import time
from typing import Any, Callable, List, Optional, Tuple


class OpenAIVideoClientService:
    def __init__(
        self,
        api_key: str,
        model_name: str = "sora-2",
        poll_interval_seconds: int = 10,
    ):
        key = str(api_key or "").strip()
        if not key:
            raise ValueError("Missing OPENAI_API_KEY configuration.")
        try:
            from openai import OpenAI
        except Exception as error:
            raise RuntimeError(
                "openai package is required for VIDEO_PROVIDER=openai. "
                "Install dependencies from requirements.txt."
            ) from error

        self.client = OpenAI(api_key=key)
        self.model_name = str(model_name or "sora-2").strip() or "sora-2"
        self.poll_interval_seconds = max(1, int(poll_interval_seconds or 10))

    def generate_video_bytes(
        self,
        prompt: str,
        reference_images: List[Any],
        aspect_ratio: Optional[str] = None,
        last_frame: Optional[Any] = None,
        allow_reference_fallback: bool = False,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> bytes:
        input_reference = self._select_input_reference(
            reference_images=reference_images,
            last_frame=last_frame,
        )
        size = self._map_size(aspect_ratio)

        create_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "prompt": str(prompt or ""),
        }
        if input_reference is not None:
            create_kwargs["input_reference"] = input_reference
        if size:
            create_kwargs["size"] = size

        try:
            video = self.client.videos.create(**create_kwargs)
        except Exception as error:
            if input_reference is not None and allow_reference_fallback:
                create_kwargs.pop("input_reference", None)
                video = self.client.videos.create(**create_kwargs)
            else:
                raise

        while str(getattr(video, "status", "")).lower() in {"queued", "in_progress"}:
            if progress_callback:
                progress = self._normalize_progress(getattr(video, "progress", None))
                if progress is None:
                    progress = 0.1
                progress_callback(progress)
            time.sleep(self.poll_interval_seconds)
            video = self.client.videos.retrieve(video.id)

        status = str(getattr(video, "status", "")).lower()
        if status == "failed":
            error_info = getattr(video, "error", None)
            raise RuntimeError(f"OpenAI video generation failed: {error_info}")
        if status != "completed":
            raise RuntimeError(f"Unexpected OpenAI video status: {status or 'unknown'}")

        content = self.client.videos.download_content(video.id, variant="video")
        if progress_callback:
            progress_callback(1.0)
        return self._content_to_bytes(content)

    def _select_input_reference(
        self,
        *,
        reference_images: List[Any],
        last_frame: Optional[Any],
    ) -> Optional[Tuple[str, bytes, str]]:
        # For continuity, prefer previous scene end-frame when available.
        if last_frame is not None:
            payload = self._image_payload_from_image_object(last_frame, default_name="last_frame")
            if payload is not None:
                return payload

        for item in reference_images or []:
            payload = self._image_payload_from_reference_item(item)
            if payload is not None:
                return payload
        return None

    def _image_payload_from_reference_item(self, item: Any) -> Optional[Tuple[str, bytes, str]]:
        image = None
        if isinstance(item, dict):
            image = item.get("image")
        else:
            image = getattr(item, "image", None) or getattr(item, "reference_image", None)
        if image is None:
            return None
        return self._image_payload_from_image_object(image, default_name="reference")

    def _image_payload_from_image_object(
        self,
        image: Any,
        *,
        default_name: str,
    ) -> Optional[Tuple[str, bytes, str]]:
        image_bytes = getattr(image, "image_bytes", None) or getattr(image, "imageBytes", None)
        mime_type = getattr(image, "mime_type", None) or getattr(image, "mimeType", None)

        data = self._as_bytes(image_bytes)
        if not data:
            return None

        mime = str(mime_type or "image/png").strip() or "image/png"
        extension = "jpg" if "jpeg" in mime else ("webp" if "webp" in mime else "png")
        filename = f"{default_name}.{extension}"
        return (filename, data, mime)

    @staticmethod
    def _as_bytes(value: Any) -> bytes:
        if value is None:
            return b""
        if isinstance(value, bytes):
            return value
        if isinstance(value, bytearray):
            return bytes(value)
        if isinstance(value, str):
            try:
                return base64.b64decode(value)
            except Exception:
                return value.encode("utf-8")
        return b""

    @staticmethod
    def _map_size(aspect_ratio: Optional[str]) -> str:
        ratio = str(aspect_ratio or "").strip()
        if ratio == "16:9":
            return "1280x720"
        if ratio == "9:16":
            return "720x1280"
        return "1280x720"

    @staticmethod
    def _normalize_progress(progress_value: Any) -> Optional[float]:
        if progress_value is None:
            return None
        try:
            numeric = float(progress_value)
        except Exception:
            return None
        if numeric > 1.0:
            numeric = numeric / 100.0
        if numeric < 0:
            numeric = 0.0
        if numeric > 1.0:
            numeric = 1.0
        return numeric

    @staticmethod
    def _content_to_bytes(content: Any) -> bytes:
        data = getattr(content, "content", None)
        if isinstance(data, bytes):
            return data
        reader = getattr(content, "read", None)
        if callable(reader):
            payload = reader()
            if isinstance(payload, bytes):
                return payload
        if isinstance(content, bytes):
            return content
        raise RuntimeError("OpenAI returned empty video content.")
