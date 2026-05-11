from __future__ import annotations

import uuid
from pathlib import Path

from flask import Blueprint, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from services.manifest_service import ManifestService
from services.local_render_runner import LocalRenderRunner
from services.validation_service import ValidationService


ALLOWED_STEPS = {
    "brief",
    "characters",
    "world",
    "scenes",
    "script",
    "audio",
    "output",
    "review",
}


def create_v2_projects_blueprint(
    manifest_service: ManifestService,
    validation_service: ValidationService,
    render_runner: LocalRenderRunner,
) -> Blueprint:
    blueprint = Blueprint("v2_projects", __name__)

    @blueprint.post("/api/v2/projects")
    def create_project():
        payload = request.get_json(silent=True) or {}
        brief = payload.get("brief", {})
        description = payload.get("description") or brief.get("description")
        seed = payload.get("seed")

        try:
            manifest = manifest_service.create_project(
                initial_brief=brief,
                seed=seed,
                description=description,
            )
        except RuntimeError as error:
            return jsonify({"error": str(error)}), 502
        completion = manifest_service.completion_state(manifest)
        return jsonify(
            {
                "project_id": manifest["project_id"],
                "seed": manifest["project_seed"],
                "manifest_preview": manifest,
                "completion": completion,
            }
        ), 201

    @blueprint.get("/api/v2/projects/<project_id>")
    def get_project(project_id: str):
        try:
            manifest = manifest_service.get_manifest(project_id)
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404
        completion = manifest_service.completion_state(manifest)
        return jsonify({"project_id": project_id, "manifest": manifest, "completion": completion})

    @blueprint.patch("/api/v2/projects/<project_id>/generation-mode")
    def set_generation_mode(project_id: str):
        payload = request.get_json(silent=True) or {}
        mode = payload.get("generation_mode", payload.get("mode", "auto"))
        try:
            manifest = manifest_service.set_generation_mode(project_id, mode)
            completion = manifest_service.completion_state(manifest)
            return jsonify(
                {
                    "project_id": project_id,
                    "generation_mode": manifest.get("generation_mode", "auto"),
                    "manifest_preview": manifest,
                    "completion": completion,
                }
            )
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

    @blueprint.patch("/api/v2/projects/<project_id>/steps/<step_name>")
    @blueprint.post("/api/v2/projects/<project_id>/steps/<step_name>")
    def update_step(project_id: str, step_name: str):
        if step_name not in ALLOWED_STEPS:
            return jsonify({"error": f"Unknown step '{step_name}'"}), 400
        payload = request.get_json(silent=True) or {}
        try:
            manifest = manifest_service.update_step(project_id, step_name, payload)
            completion = manifest_service.completion_state(manifest)
            return jsonify({"manifest_preview": manifest, "completion": completion})
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except RuntimeError as error:
            return jsonify({"error": str(error)}), 502

    @blueprint.post("/api/v2/projects/<project_id>/steps/<step_name>/skip")
    def skip_step(project_id: str, step_name: str):
        if step_name not in ALLOWED_STEPS:
            return jsonify({"error": f"Unknown step '{step_name}'"}), 400
        try:
            manifest = manifest_service.skip_step(project_id, step_name)
            completion = manifest_service.completion_state(manifest)
            return jsonify({"manifest_preview": manifest, "completion": completion})
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except RuntimeError as error:
            return jsonify({"error": str(error)}), 502

    @blueprint.post("/api/v2/projects/<project_id>/steps/<step_name>/surprise")
    def surprise_step(project_id: str, step_name: str):
        if step_name not in ALLOWED_STEPS:
            return jsonify({"error": f"Unknown step '{step_name}'"}), 400
        try:
            manifest = manifest_service.surprise_step(project_id, step_name)
            completion = manifest_service.completion_state(manifest)
            return jsonify({"manifest_preview": manifest, "completion": completion})
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except RuntimeError as error:
            return jsonify({"error": str(error)}), 502

    @blueprint.post("/api/v2/projects/<project_id>/scenes/<scene_id>/lock")
    def lock_scene(project_id: str, scene_id: str):
        payload = request.get_json(silent=True) or {}
        locked = bool(payload.get("locked", True))
        try:
            manifest = manifest_service.lock_scene(project_id, scene_id, locked)
            completion = manifest_service.completion_state(manifest)
            return jsonify(
                {
                    "manifest_preview": manifest,
                    "completion": completion,
                    "scene_id": scene_id,
                    "locked": locked,
                }
            )
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404

    @blueprint.post("/api/v2/projects/<project_id>/scenes/<scene_id>/approve")
    def approve_scene(project_id: str, scene_id: str):
        payload = request.get_json(silent=True) or {}
        approved = bool(payload.get("approved", True))
        note = str(payload.get("note", "")).strip()
        try:
            manifest = manifest_service.set_scene_approval(
                project_id,
                scene_id,
                approved=approved,
                note=note,
            )
            completion = manifest_service.completion_state(manifest)
            return jsonify(
                {
                    "manifest_preview": manifest,
                    "completion": completion,
                    "scene_id": scene_id,
                    "approved": approved,
                }
            )
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

    @blueprint.patch("/api/v2/projects/<project_id>/scenes/<scene_id>/state")
    def patch_scene_state(project_id: str, scene_id: str):
        payload = request.get_json(silent=True) or {}
        state = payload.get("state")
        note = str(payload.get("note", "")).strip()
        if not state:
            return jsonify({"error": "state is required"}), 400
        try:
            manifest = manifest_service.set_scene_state(
                project_id,
                scene_id,
                state,
                note=note,
            )
            completion = manifest_service.completion_state(manifest)
            return jsonify(
                {
                    "manifest_preview": manifest,
                    "completion": completion,
                    "scene_id": scene_id,
                    "state": state,
                }
            )
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

    @blueprint.patch("/api/v2/projects/<project_id>/scenes/<scene_id>")
    def patch_scene_content(project_id: str, scene_id: str):
        payload = request.get_json(silent=True) or {}
        try:
            manifest = manifest_service.update_scene_content(project_id, scene_id, payload)
            completion = manifest_service.completion_state(manifest)
            return jsonify(
                {
                    "manifest_preview": manifest,
                    "completion": completion,
                    "scene_id": scene_id,
                }
            )
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

    @blueprint.patch("/api/v2/projects/<project_id>/scenes/<scene_id>/trim")
    def patch_scene_trim(project_id: str, scene_id: str):
        payload = request.get_json(silent=True) or {}
        try:
            start_trim = float(payload.get("start_trim", 0.0))
            end_trim = float(payload.get("end_trim", 0.0))
        except (TypeError, ValueError):
            return jsonify({"error": "start_trim and end_trim must be numeric"}), 400
        try:
            manifest = manifest_service.update_scene_trim(
                project_id,
                scene_id,
                start_trim=start_trim,
                end_trim=end_trim,
            )
            completion = manifest_service.completion_state(manifest)
            return jsonify(
                {
                    "manifest_preview": manifest,
                    "completion": completion,
                    "scene_id": scene_id,
                    "trim": {"start_trim": start_trim, "end_trim": end_trim},
                }
            )
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

    @blueprint.patch(
        "/api/v2/projects/<project_id>/scenes/<scene_id>/dialogue/<int:line_index>"
    )
    def patch_scene_dialogue_line(project_id: str, scene_id: str, line_index: int):
        payload = request.get_json(silent=True) or {}
        try:
            manifest = manifest_service.update_subtitle_line(
                project_id,
                scene_id,
                line_index=line_index,
                payload=payload,
            )
            completion = manifest_service.completion_state(manifest)
            return jsonify(
                {
                    "manifest_preview": manifest,
                    "completion": completion,
                    "scene_id": scene_id,
                    "line_index": line_index,
                }
            )
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404
        except IndexError as error:
            return jsonify({"error": str(error)}), 400
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

    @blueprint.post("/api/v2/projects/<project_id>/scenes/<scene_id>/regenerate")
    def regenerate_scene(project_id: str, scene_id: str):
        payload = request.get_json(silent=True) or {}
        branch = bool(payload.get("branch", False))
        activate = bool(payload.get("activate", True))
        try:
            manifest = manifest_service.regenerate_scene(
                project_id,
                scene_id,
                branch=branch,
                activate=activate,
            )
            completion = manifest_service.completion_state(manifest)
            return jsonify(
                {
                    "manifest_preview": manifest,
                    "completion": completion,
                    "scene_id": scene_id,
                    "branch": branch,
                    "activate": activate,
                }
            )
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

    @blueprint.post("/api/v2/projects/<project_id>/scenes/<scene_id>/regenerate-video")
    def regenerate_scene_video(project_id: str, scene_id: str):
        try:
            manifest = manifest_service.get_manifest(project_id)
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404

        scene_exists = any(
            isinstance(scene, dict) and scene.get("scene_id") == scene_id
            for scene in manifest.get("scenes", [])
        )
        if not scene_exists:
            return jsonify({"error": f"Scene '{scene_id}' not found."}), 404

        job = manifest_service.create_job(project_id)
        try:
            manifest_service.update_job(
                project_id,
                job["job_id"],
                status="queued",
                progress=0.01,
                current_stage="queued_scene_regen",
                current_scene_id=scene_id,
            )
            render_runner.start_with_options(
                project_id,
                job["job_id"],
                target_scene_id=scene_id,
                scene_only=True,
            )
            return jsonify(
                {
                    "job_id": job["job_id"],
                    "project_id": project_id,
                    "scene_id": scene_id,
                    "runner": "local_thread",
                    "status": "queued",
                    "mode": "scene_only",
                }
            )
        except ValueError as error:
            manifest_service.update_job(
                project_id,
                job["job_id"],
                status="failed",
                append_error=str(error),
            )
            return jsonify({"error": str(error)}), 409
        except Exception as error:
            manifest_service.update_job(
                project_id,
                job["job_id"],
                status="failed",
                append_error=str(error),
            )
            return jsonify({"error": f"Failed to enqueue scene regeneration: {error}"}), 500

    @blueprint.post(
        "/api/v2/projects/<project_id>/scenes/<scene_id>/variants/<variant_id>/activate"
    )
    def activate_variant(project_id: str, scene_id: str, variant_id: str):
        try:
            manifest = manifest_service.activate_scene_variant(
                project_id,
                scene_id,
                variant_id,
            )
            completion = manifest_service.completion_state(manifest)
            return jsonify(
                {
                    "manifest_preview": manifest,
                    "completion": completion,
                    "scene_id": scene_id,
                    "variant_id": variant_id,
                }
            )
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404

    @blueprint.post("/api/v2/projects/<project_id>/approvals/script")
    def approve_script(project_id: str):
        payload = request.get_json(silent=True) or {}
        approved = bool(payload.get("approved", False))
        note = str(payload.get("note", "")).strip()
        try:
            manifest = manifest_service.set_script_approval(
                project_id,
                approved=approved,
                reviewer_note=note,
            )
            completion = manifest_service.completion_state(manifest)
            return jsonify(
                {
                    "manifest_preview": manifest,
                    "completion": completion,
                    "approved": approved,
                }
            )
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404

    @blueprint.get("/api/v2/projects/<project_id>/timeline")
    def get_timeline(project_id: str):
        try:
            manifest = manifest_service.get_manifest(project_id)
            return jsonify({"project_id": project_id, "timeline": manifest.get("timeline", {})})
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404

    @blueprint.patch("/api/v2/projects/<project_id>/timeline")
    def patch_timeline(project_id: str):
        payload = request.get_json(silent=True) or {}
        timeline = payload.get("timeline", payload)
        if not isinstance(timeline, dict):
            return jsonify({"error": "timeline payload must be a JSON object"}), 400
        try:
            manifest = manifest_service.update_timeline(project_id, timeline)
            completion = manifest_service.completion_state(manifest)
            return jsonify({"manifest_preview": manifest, "completion": completion})
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404

    @blueprint.get("/api/v2/projects/<project_id>/character-bible")
    def get_character_bible(project_id: str):
        try:
            manifest = manifest_service.get_manifest(project_id)
            return jsonify(
                {
                    "project_id": project_id,
                    "character_bible": manifest.get("character_bible", {}),
                }
            )
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404

    @blueprint.patch("/api/v2/projects/<project_id>/character-bible")
    def patch_character_bible(project_id: str):
        payload = request.get_json(silent=True) or {}
        bible_payload = payload.get("character_bible", payload)
        if not isinstance(bible_payload, dict):
            return jsonify({"error": "character_bible payload must be a JSON object"}), 400
        try:
            manifest = manifest_service.update_character_bible(project_id, bible_payload)
            completion = manifest_service.completion_state(manifest)
            return jsonify({"manifest_preview": manifest, "completion": completion})
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404

    @blueprint.route(
        "/api/v2/projects/<project_id>/characters/<character_id>/images",
        methods=["POST", "PATCH"],
        strict_slashes=False,
    )
    def upload_character_images(project_id: str, character_id: str):
        allowed_ext = {".png", ".jpg", ".jpeg", ".webp"}
        files = request.files.getlist("images")
        if not files and "image" in request.files:
            files = [request.files["image"]]
        if not files:
            return jsonify({"error": "No image files provided. Use form-data field 'images'."}), 400

        try:
            manifest_service.get_manifest(project_id)
            paths = manifest_service.project_paths(project_id)
            project_dir = Path(paths["project_dir"])
            target_dir = Path(paths["assets_reference_dir"]) / "characters" / character_id
            target_dir.mkdir(parents=True, exist_ok=True)
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404

        uploaded = []
        for item in files:
            if not item or not item.filename:
                continue
            filename = secure_filename(item.filename)
            extension = Path(filename).suffix.lower()
            if extension not in allowed_ext:
                return (
                    jsonify(
                        {
                            "error": (
                                "Unsupported image type. "
                                "Allowed: .png, .jpg, .jpeg, .webp"
                            )
                        }
                    ),
                    400,
                )
            final_name = f"{character_id}_{uuid.uuid4().hex[:10]}{extension}"
            final_path = target_dir / final_name
            item.save(final_path)
            relative = str(final_path.relative_to(project_dir)).replace("\\", "/")
            uploaded.append(relative)

        if not uploaded:
            return jsonify({"error": "No valid files were uploaded."}), 400

        try:
            manifest = None
            for relative_path in uploaded:
                manifest = manifest_service.attach_character_reference_image(
                    project_id,
                    character_id,
                    relative_path,
                )
            completion = manifest_service.completion_state(manifest or {})
            return jsonify(
                {
                    "project_id": project_id,
                    "character_id": character_id,
                    "uploaded_images": [
                        {
                            "path": path,
                            "url": f"/api/v2/projects/{project_id}/files/{path}",
                        }
                        for path in uploaded
                    ],
                    "manifest_preview": manifest,
                    "completion": completion,
                }
            ), 201
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404
        except Exception as error:
            return jsonify({"error": f"Image upload failed: {error}"}), 500

    @blueprint.get("/api/v2/projects/<project_id>/estimate")
    def estimate_project(project_id: str):
        try:
            estimate = manifest_service.estimate_project(project_id)
            return jsonify(estimate)
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404

    @blueprint.get("/api/v2/projects/<project_id>/snapshots")
    def list_snapshots(project_id: str):
        try:
            snapshots = manifest_service.list_snapshots(project_id)
            return jsonify({"project_id": project_id, "snapshots": snapshots})
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404

    @blueprint.post("/api/v2/projects/<project_id>/snapshots")
    def create_snapshot(project_id: str):
        payload = request.get_json(silent=True) or {}
        label = str(payload.get("label", "manual")).strip() or "manual"
        try:
            snapshot = manifest_service.create_snapshot(project_id, label=label)
            return jsonify(snapshot), 201
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404

    @blueprint.post("/api/v2/projects/<project_id>/snapshots/<snapshot_id>/rollback")
    def rollback_snapshot(project_id: str, snapshot_id: str):
        try:
            manifest = manifest_service.rollback_snapshot(project_id, snapshot_id)
            completion = manifest_service.completion_state(manifest)
            return jsonify(
                {
                    "project_id": project_id,
                    "snapshot_id": snapshot_id,
                    "manifest_preview": manifest,
                    "completion": completion,
                }
            )
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

    @blueprint.post("/api/v2/projects/<project_id>/render")
    def render_project(project_id: str):
        try:
            manifest = manifest_service.get_manifest(project_id)
            manifest = validation_service.normalize_manifest(manifest)
            missing = validation_service.validate_brief_required(manifest)
            if missing:
                return (
                    jsonify(
                        {
                            "error": "Cannot render yet.",
                            "missing_required_brief_fields": missing,
                        }
                    ),
                    400,
                )
            validation_service.ensure_render_eligible(manifest)
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404
        except Exception as error:
            return jsonify({"error": str(error)}), 400

        try:
            manifest_service.create_snapshot(project_id, label="pre_render")
        except Exception:
            # Snapshot is best-effort and should not block render job enqueue.
            pass

        job = manifest_service.create_job(project_id)
        try:
            manifest_service.update_job(
                project_id,
                job["job_id"],
                status="queued",
                progress=0.01,
                current_stage="queued",
            )
            render_runner.start(project_id, job["job_id"])
            return jsonify(
                {
                    "job_id": job["job_id"],
                    "project_id": project_id,
                    "task_id": None,
                    "runner": "local_thread",
                    "status": "queued",
                }
            )
        except ValueError as error:
            manifest_service.update_job(
                project_id,
                job["job_id"],
                status="failed",
                append_error=str(error),
            )
            return jsonify({"error": str(error)}), 409
        except Exception as error:
            manifest_service.update_job(
                project_id,
                job["job_id"],
                status="failed",
                append_error=str(error),
            )
            return jsonify({"error": f"Failed to enqueue render job: {error}"}), 500

    @blueprint.post("/api/v2/projects/<project_id>/export")
    def export_bundle(project_id: str):
        try:
            artifact = manifest_service.create_export_bundle(project_id)
            return jsonify({"project_id": project_id, "bundle": artifact}), 201
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404

    @blueprint.get("/api/v2/projects/<project_id>/artifacts")
    def list_artifacts(project_id: str):
        try:
            artifacts = manifest_service.list_artifacts(project_id)
            return jsonify({"project_id": project_id, "artifacts": artifacts})
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404

    @blueprint.get("/api/v2/projects/<project_id>/files/<path:relative_path>")
    def serve_project_file(project_id: str, relative_path: str):
        try:
            parent, filename = manifest_service.get_project_file(project_id, relative_path)
            return send_from_directory(parent, filename, as_attachment=False)
        except FileNotFoundError:
            return jsonify({"error": "File not found"}), 404
        except PermissionError:
            return jsonify({"error": "Invalid file path"}), 400

    return blueprint
