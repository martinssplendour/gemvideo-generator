from __future__ import annotations

from flask import Blueprint, jsonify, request

from services.manifest_service import ManifestService
from services.voice_service import VoiceService


def create_v2_voices_blueprint(
    manifest_service: ManifestService,
    voice_service: VoiceService,
) -> Blueprint:
    blueprint = Blueprint("v2_voices", __name__)

    @blueprint.get("/api/v2/voices")
    def list_voices():
        return jsonify(voice_service.list_voices())

    @blueprint.patch("/api/v2/projects/<project_id>/characters/<character_id>/voice")
    def assign_voice(project_id: str, character_id: str):
        payload = request.get_json(silent=True) or {}
        provider = payload.get("voice_provider", "native")
        voice_id = payload.get("voice_id")
        speaking_style = payload.get("speaking_style", "clear, expressive, medium pace")

        try:
            manifest = manifest_service.get_manifest(project_id)
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404

        target = None
        for character in manifest.get("characters", []):
            if character.get("character_id") == character_id:
                target = character
                break

        if target is None:
            return jsonify({"error": f"Character '{character_id}' not found."}), 404

        if not voice_id:
            default_voice = voice_service.assign_default_voice(
                project_seed=manifest.get("project_seed", ""),
                character_id=character_id,
                preferred_provider=provider,
            )
            provider = default_voice["voice_provider"]
            voice_id = default_voice["voice_id"]
            speaking_style = default_voice["speaking_style"]

        target["voice"] = {
            "voice_provider": provider,
            "voice_id": voice_id,
            "speaking_style": speaking_style,
        }
        manifest_service.save_manifest(project_id, manifest)

        return jsonify(
            {
                "project_id": project_id,
                "character_id": character_id,
                "voice": target["voice"],
                "manifest_preview": manifest,
            }
        )

    return blueprint
