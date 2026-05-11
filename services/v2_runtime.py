from __future__ import annotations

from dataclasses import dataclass

from google import genai

from config import AppSettings
from services.autofill_service import ManifestAutoFillService
from services.description_planner_service import DescriptionPlannerService
from services.manifest_service import ManifestService
from services.openai_video_client_service import OpenAIVideoClientService
from services.render_service import RenderService
from services.report_service import ReportService
from services.safety_service import GeminiSafetyClassifier, SafetyCheckService
from services.scene_builder_service import SceneBuilderService
from services.validation_service import ValidationService
from services.veo_client_service import VeoClientService
from services.voice_service import VoiceService


@dataclass
class V2Services:
    validation_service: ValidationService
    scene_builder_service: SceneBuilderService
    autofill_service: ManifestAutoFillService
    manifest_service: ManifestService
    safety_service: SafetyCheckService
    voice_service: VoiceService
    report_service: ReportService
    render_service: RenderService


def build_v2_services(settings: AppSettings) -> V2Services:
    validation_service = ValidationService()
    scene_builder_service = SceneBuilderService(
        api_key=settings.api_key,
        model_name=settings.scene_builder_model,
        timeout_seconds=settings.scene_builder_timeout_seconds,
        enabled=settings.enable_scene_builder_model,
    )
    autofill_service = ManifestAutoFillService(
        scene_builder_service=scene_builder_service
    )
    description_planner_service = DescriptionPlannerService(
        api_key=settings.api_key,
        model_name=settings.manifest_planner_model,
        timeout_seconds=settings.manifest_planner_timeout_seconds,
        enabled=settings.enable_manifest_planner,
    )
    manifest_service = ManifestService(
        data_dir=settings.data_dir,
        validation_service=validation_service,
        autofill_service=autofill_service,
        description_planner_service=description_planner_service,
    )
    model_classifier = None
    if settings.enable_model_safety and settings.api_key:
        safety_client = genai.Client(api_key=settings.api_key)
        model_classifier = GeminiSafetyClassifier(
            client=safety_client,
            model_name=settings.safety_model_name,
        )
    safety_service = SafetyCheckService(
        profile=settings.safety_profile,
        mode=settings.safety_mode,
        model_classifier=model_classifier,
    )
    voice_service = VoiceService(
        elevenlabs_api_key=settings.elevenlabs_api_key,
        elevenlabs_model_id=settings.elevenlabs_model_id,
        native_tts_engine=settings.native_tts_engine,
    )
    report_service = ReportService()
    if settings.video_provider == "openai":
        video_client_service = OpenAIVideoClientService(
            api_key=settings.openai_api_key,
            model_name=settings.openai_video_model,
            poll_interval_seconds=settings.poll_interval_seconds,
        )
    else:
        video_client_service = VeoClientService(
            api_key=settings.api_key,
            model_name=settings.model_name,
            poll_interval_seconds=settings.poll_interval_seconds,
        )
    render_service = RenderService(
        manifest_service=manifest_service,
        validation_service=validation_service,
        autofill_service=autofill_service,
        safety_service=safety_service,
        voice_service=voice_service,
        report_service=report_service,
        veo_client_service=video_client_service,
        enable_real_generation=settings.enable_real_generation,
        pause_poll_seconds=settings.pause_poll_seconds,
    )
    return V2Services(
        validation_service=validation_service,
        scene_builder_service=scene_builder_service,
        autofill_service=autofill_service,
        manifest_service=manifest_service,
        safety_service=safety_service,
        voice_service=voice_service,
        report_service=report_service,
        render_service=render_service,
    )
