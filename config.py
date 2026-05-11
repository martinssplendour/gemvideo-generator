import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppSettings:
    api_key: str
    video_provider: str = "gemini"
    output_dir: str = "output"
    upload_dir: str = "uploads"
    data_dir: str = "data"
    static_folder: str = "."
    model_name: str = "veo-3.1-generate-preview"
    openai_api_key: str = ""
    openai_video_model: str = "sora-2"
    max_retries: int = 3
    retry_delay_seconds: int = 5
    poll_interval_seconds: int = 10
    elevenlabs_api_key: str = ""
    elevenlabs_model_id: str = "eleven_multilingual_v2"
    native_tts_engine: str = "pyttsx3"
    enable_real_generation: bool = True
    safety_mode: str = "warn_only"
    safety_profile: str = "standard"
    enable_model_safety: bool = False
    safety_model_name: str = "gemini-2.5-flash"
    pause_poll_seconds: float = 1.0
    app_base_url: str = "http://localhost:5000"
    enable_manifest_planner: bool = True
    manifest_planner_model: str = "gemini-2.5-flash"
    manifest_planner_timeout_seconds: int = 12
    enable_scene_builder_model: bool = True
    scene_builder_model: str = "gemini-2.5-flash"
    scene_builder_timeout_seconds: int = 18


def load_settings() -> AppSettings:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    video_provider = os.getenv("VIDEO_PROVIDER", "gemini").strip().lower() or "gemini"
    if video_provider not in {"gemini", "openai"}:
        raise ValueError("VIDEO_PROVIDER must be either 'gemini' or 'openai'.")
    if video_provider == "gemini" and not api_key:
        raise ValueError("Missing GEMINI_API_KEY configuration for VIDEO_PROVIDER=gemini.")
    if video_provider == "openai" and not openai_api_key:
        raise ValueError("Missing OPENAI_API_KEY configuration for VIDEO_PROVIDER=openai.")
    if not api_key and not openai_api_key:
        raise ValueError("Provide at least one key: GEMINI_API_KEY or OPENAI_API_KEY.")
    return AppSettings(
        api_key=api_key,
        video_provider=video_provider,
        output_dir=os.getenv("OUTPUT_DIR", "output"),
        upload_dir=os.getenv("UPLOAD_DIR", "uploads"),
        data_dir=os.getenv("DATA_DIR", "data"),
        static_folder=os.getenv("STATIC_FOLDER", "."),
        model_name=os.getenv("VEO_MODEL_NAME", "veo-3.1-generate-preview"),
        openai_api_key=openai_api_key,
        openai_video_model=os.getenv("OPENAI_VIDEO_MODEL", "sora-2").strip(),
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        retry_delay_seconds=int(os.getenv("RETRY_DELAY_SECONDS", "5")),
        poll_interval_seconds=int(os.getenv("POLL_INTERVAL_SECONDS", "10")),
        elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY", "").strip(),
        elevenlabs_model_id=os.getenv(
            "ELEVENLABS_MODEL_ID", "eleven_multilingual_v2"
        ).strip(),
        native_tts_engine=os.getenv("NATIVE_TTS_ENGINE", "pyttsx3").strip(),
        enable_real_generation=os.getenv("ENABLE_REAL_GENERATION", "true").lower()
        in ("1", "true", "yes"),
        safety_mode=os.getenv("SAFETY_MODE", "warn_only"),
        safety_profile=os.getenv("SAFETY_PROFILE", "standard"),
        enable_model_safety=os.getenv("ENABLE_MODEL_SAFETY", "false").lower()
        in ("1", "true", "yes"),
        safety_model_name=os.getenv("SAFETY_MODEL_NAME", "gemini-2.5-flash"),
        pause_poll_seconds=float(os.getenv("PAUSE_POLL_SECONDS", "1.0")),
        app_base_url=os.getenv("APP_BASE_URL", "http://localhost:5000"),
        enable_manifest_planner=os.getenv("ENABLE_MANIFEST_PLANNER", "true").lower()
        in ("1", "true", "yes"),
        manifest_planner_model=os.getenv(
            "MANIFEST_PLANNER_MODEL", "gemini-2.5-flash"
        ).strip(),
        manifest_planner_timeout_seconds=int(
            os.getenv("MANIFEST_PLANNER_TIMEOUT_SECONDS", "12")
        ),
        enable_scene_builder_model=os.getenv(
            "ENABLE_SCENE_BUILDER_MODEL", "true"
        ).lower()
        in ("1", "true", "yes"),
        scene_builder_model=os.getenv(
            "SCENE_BUILDER_MODEL", "gemini-2.5-flash"
        ).strip(),
        scene_builder_timeout_seconds=int(
            os.getenv("SCENE_BUILDER_TIMEOUT_SECONDS", "18")
        ),
    )
