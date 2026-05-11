from __future__ import annotations

import shutil

from flask import Blueprint, jsonify

from config import AppSettings
from services.manifest_service import ManifestService


def create_v2_jobs_blueprint(
    manifest_service: ManifestService,
    settings: AppSettings,
) -> Blueprint:
    blueprint = Blueprint("v2_jobs", __name__)

    @blueprint.get("/api/v2/jobs/<job_id>")
    def get_job(job_id: str):
        try:
            job = manifest_service.get_job(job_id)
            return jsonify(job)
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404

    @blueprint.post("/api/v2/jobs/<job_id>/pause")
    def pause_job(job_id: str):
        try:
            job = manifest_service.request_job_action(job_id, "pause")
            return jsonify(job)
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

    @blueprint.post("/api/v2/jobs/<job_id>/resume")
    def resume_job(job_id: str):
        try:
            job = manifest_service.request_job_action(job_id, "resume")
            return jsonify(job)
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

    @blueprint.post("/api/v2/jobs/<job_id>/cancel")
    def cancel_job(job_id: str):
        try:
            job = manifest_service.request_job_action(job_id, "cancel")
            return jsonify(job)
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

    @blueprint.get("/api/v2/health")
    def health():
        ffmpeg_path = shutil.which("ffmpeg")
        return jsonify(
            {
                "status": "ok",
                "services": {
                    "ffmpeg": {
                        "available": ffmpeg_path is not None,
                        "path": ffmpeg_path or "",
                    },
                    "render_runner": {
                        "mode": "local_thread",
                        "queue": "in_process",
                    },
                    "video_provider": settings.video_provider,
                    "video_model": (
                        settings.openai_video_model
                        if settings.video_provider == "openai"
                        else settings.model_name
                    ),
                    "safety_profile": settings.safety_profile,
                },
            }
        )

    return blueprint
