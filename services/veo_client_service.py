import time
from typing import Any, Callable, List, Optional

from google import genai
from google.genai import types


class VeoClientService:
    def __init__(self, api_key: str, model_name: str, poll_interval_seconds: int = 10):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.poll_interval_seconds = poll_interval_seconds

    def generate_video_bytes(
        self,
        prompt: str,
        reference_images: List[Any],
        aspect_ratio: Optional[str] = None,
        last_frame: Optional[types.Image] = None,
        allow_reference_fallback: bool = False,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> bytes:
        def run_generation(
            refs: List[Any] | None = None,
            frame: Optional[types.Image] = None,
        ):
            config = types.GenerateVideosConfig()
            if refs:
                config.reference_images = refs
            if aspect_ratio:
                config.aspect_ratio = aspect_ratio
            if frame is not None:
                config.last_frame = frame
            return self.client.models.generate_videos(
                model=self.model_name,
                prompt=prompt,
                config=config,
            )

        try:
            operation = run_generation(reference_images, last_frame)
        except Exception as error:
            message = str(error)
            if last_frame is not None and self._is_unsupported_request_error(message):
                # Some Veo model versions reject last_frame continuation requests.
                operation = run_generation(reference_images, None)
            elif reference_images and self._is_reference_validation_error(message):
                if allow_reference_fallback:
                    operation = run_generation([], last_frame)
                else:
                    raise RuntimeError(
                        "Reference images were rejected by Veo request validation. "
                        f"Details: {message}. "
                        "Aborting instead of silently dropping references."
                    ) from error
            else:
                raise

        print("Generation started. Polling...")
        poll_count = 0
        while not operation.done:
            poll_count += 1
            if progress_callback:
                progress_value = self._operation_progress(operation)
                if progress_value is None:
                    progress_value = min(0.9, 0.08 + (poll_count * 0.08))
                progress_callback(progress_value)
            print(".", end="", flush=True)
            time.sleep(self.poll_interval_seconds)
            operation = self.client.operations.get(operation=operation)
        print("\nDone.")

        if operation.error:
            error_message = str(operation.error)
            if last_frame is not None and self._is_unsupported_request_error(error_message):
                operation = run_generation(reference_images, None)
                print("Generation started (retry without last_frame). Polling...")
                poll_count = 0
                while not operation.done:
                    poll_count += 1
                    if progress_callback:
                        progress_value = self._operation_progress(operation)
                        if progress_value is None:
                            progress_value = min(0.9, 0.08 + (poll_count * 0.08))
                        progress_callback(progress_value)
                    print(".", end="", flush=True)
                    time.sleep(self.poll_interval_seconds)
                    operation = self.client.operations.get(operation=operation)
                print("\nDone.")
                if operation.error:
                    raise RuntimeError(f"API error: {operation.error}")
            elif reference_images and self._is_reference_validation_error(error_message):
                if not allow_reference_fallback:
                    raise RuntimeError(
                        "Veo rejected reference images during generation. "
                        f"Details: {error_message}. "
                        "Generation stopped to avoid identity drift."
                    )
                operation = run_generation([], last_frame)
                print("Generation started (retry without references). Polling...")
                poll_count = 0
                while not operation.done:
                    poll_count += 1
                    if progress_callback:
                        progress_value = self._operation_progress(operation)
                        if progress_value is None:
                            progress_value = min(0.9, 0.08 + (poll_count * 0.08))
                        progress_callback(progress_value)
                    print(".", end="", flush=True)
                    time.sleep(self.poll_interval_seconds)
                    operation = self.client.operations.get(operation=operation)
                print("\nDone.")
                if operation.error:
                    raise RuntimeError(f"API error: {operation.error}")
            else:
                raise RuntimeError(f"API error: {operation.error}")

        if not operation.response or not operation.response.generated_videos:
            raise RuntimeError("Generation finished, but no video was returned.")
        if progress_callback:
            progress_callback(1.0)

        video_resource = operation.response.generated_videos[0].video
        return self.client.files.download(file=video_resource)

    def _operation_progress(self, operation) -> Optional[float]:
        metadata = getattr(operation, "metadata", None)
        if metadata is None:
            return None

        candidates = ["progress_percent", "progress", "progressPercent", "percent_complete"]
        for key in candidates:
            value = None
            if isinstance(metadata, dict):
                value = metadata.get(key)
            else:
                value = getattr(metadata, key, None)
            if value is None:
                continue
            try:
                numeric = float(value)
            except Exception:
                continue
            if numeric > 1.0:
                numeric = numeric / 100.0
            if numeric < 0:
                numeric = 0.0
            if numeric > 1:
                numeric = 1.0
            return numeric
        return None

    @staticmethod
    def _is_reference_validation_error(message: str) -> bool:
        lowered = (message or "").lower()
        if "image can not be found in the request" in lowered:
            return True
        if "reference image" in lowered:
            return True
        if "reference_images" in lowered:
            return True
        if "referenceimages" in lowered:
            return True
        return False

    @staticmethod
    def _is_unsupported_request_error(message: str) -> bool:
        lowered = (message or "").lower()
        if "unsupported video generation request" in lowered:
            return True
        if "unsupported request" in lowered and "video" in lowered:
            return True
        if "invalid_argument" in lowered and "supported usage" in lowered:
            return True
        return False
