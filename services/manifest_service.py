from __future__ import annotations

import copy
import json
import os
import re
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from schemas import REQUIRED_BRIEF_FIELDS, utc_now_iso
from services.autofill_service import ManifestAutoFillService
from services.description_planner_service import DescriptionPlannerService
from services.validation_service import ValidationService


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ManifestService:
    SCENE_STATES = {"planned", "generated", "approved", "locked", "stitched"}
    SCENE_TYPES = {
        "auto",
        "hook",
        "problem",
        "reframe",
        "solution",
        "payoff",
        "cta",
        "b_roll",
        "testimonial",
        "tutorial",
        "transition",
        "custom",
    }
    SCENE_TRANSITIONS = {
        "planned": {"generated", "approved", "locked"},
        "generated": {"approved", "planned", "locked"},
        "approved": {"locked", "generated", "stitched"},
        "locked": {"approved", "generated"},
        "stitched": {"approved", "locked"},
    }
    _io_lock = threading.RLock()

    def __init__(
        self,
        data_dir: str,
        validation_service: ValidationService,
        autofill_service: ManifestAutoFillService,
        description_planner_service: Optional[DescriptionPlannerService] = None,
    ):
        self.data_dir = Path(data_dir)
        self.projects_root = self.data_dir / "projects"
        self.jobs_index_path = self.data_dir / "jobs_index.json"
        self.validation_service = validation_service
        self.autofill_service = autofill_service
        self.description_planner_service = description_planner_service

        self.projects_root.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not self.jobs_index_path.exists():
            self._write_json(self.jobs_index_path, {})

    def create_project(
        self,
        initial_brief: Optional[Dict[str, Any]] = None,
        seed: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        project_id = uuid.uuid4().hex[:12]
        now = _utc_now()
        manifest = {
            "project_id": project_id,
            "created_at": now,
            "updated_at": now,
            "project_seed": seed or uuid.uuid4().hex,
            "surprise_counters": {},
            "generation_mode": "auto",
            "brief": initial_brief or {},
            "characters": [],
            "world": {},
            "scenes": [],
            "output": {
                "resolution": "1080p",
                "fps": 30,
                "subtitles_enabled": True,
                "subtitle_style": "clean lower-third",
                "safe_areas": "social-safe margins",
                "file_format": "mp4",
                "deliverables": [
                    "final_video",
                    "script",
                    "scene_list",
                    "assets_list",
                ],
            },
            "audio": {
                "music_mood": "",
                "sound_effects": [],
                "narrator_track": None,
            },
            "timeline": [],
            "safety_warnings": [],
            "autofill_log": [],
            "scene_variants": {},
            "character_bible": {},
            "render_iteration": 0,
            "approvals": {
                "script_approved": False,
                "script_needs_user_approval": True,
                "script_last_reviewed_at": None,
            },
        }
        manifest = self.validation_service.normalize_manifest(manifest)
        description_text = str(description or "").strip()
        if not description_text:
            description_text = str(manifest.get("brief", {}).get("description", "")).strip()

        if description_text:
            force_scene_rebuild = self._should_force_scene_builder_from_description(
                manifest,
                description_text,
            )
            self._apply_description_to_manifest(
                manifest,
                description_text,
                reason="description_bootstrap",
            )
            manifest = self.autofill_service.autofill_all(
                manifest,
                reason="description_bootstrap",
                force_scene_rebuild=force_scene_rebuild,
            )
            self.autofill_service.sync_character_bible(
                manifest,
                reason="description_bootstrap",
            )
        else:
            self.autofill_service.sync_character_bible(manifest, reason="project_init")
        manifest = self.validation_service.normalize_manifest(manifest)
        self.save_manifest(project_id, manifest)
        self._write_json(self._autofill_log_path(project_id), manifest["autofill_log"])
        return manifest

    def get_manifest(self, project_id: str) -> Dict[str, Any]:
        path = self._manifest_path(project_id)
        if not path.exists():
            raise FileNotFoundError(f"Project '{project_id}' not found.")
        manifest = self._read_json(path)
        return self.validation_service.normalize_manifest(manifest)

    def save_manifest(self, project_id: str, manifest: Dict[str, Any]) -> None:
        project_dir = self._project_dir(project_id)
        project_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_project_layout(project_id)
        manifest["updated_at"] = utc_now_iso()
        self._write_json(self._manifest_path(project_id), manifest)
        self._write_json(self._autofill_log_path(project_id), manifest.get("autofill_log", []))

    def _should_force_scene_builder_from_description(
        self,
        manifest: Dict[str, Any],
        description_text: str,
    ) -> bool:
        if not str(description_text or "").strip():
            return False
        scene_builder = getattr(self.autofill_service, "scene_builder_service", None)
        return bool(scene_builder and getattr(scene_builder, "enabled", False))

    def update_step(
        self, project_id: str, step_name: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        manifest = self.get_manifest(project_id)

        if step_name == "brief":
            brief_payload = payload.get("brief", payload)
            if isinstance(brief_payload, dict):
                manifest.setdefault("brief", {}).update(brief_payload)
            description_text = str(
                payload.get("description")
                or manifest.get("brief", {}).get("description", "")
            ).strip()
            should_generate_full = payload.get("generate_full_manifest")
            if should_generate_full is None:
                should_generate_full = bool(description_text)
            force_scene_rebuild = self._should_force_scene_builder_from_description(
                manifest,
                description_text,
            )
            if description_text:
                self._apply_description_to_manifest(
                    manifest,
                    description_text,
                    reason="brief_description_update",
                )
            if should_generate_full:
                manifest = self.autofill_service.autofill_all(
                    manifest,
                    reason="brief_description_update",
                    force_scene_rebuild=force_scene_rebuild,
                )
                self.autofill_service.sync_character_bible(
                    manifest,
                    reason="brief_description_update",
                )
        elif step_name == "characters":
            characters = payload.get("characters", payload)
            manifest["characters"] = (
                self._normalize_character_entries(characters)
                if isinstance(characters, list)
                else []
            )
            self.autofill_service.sync_character_bible(manifest, reason="manual_characters")
        elif step_name == "world":
            world_payload = payload.get("world", payload)
            if isinstance(world_payload, dict):
                merged = dict(world_payload)
                visual_references_text = merged.pop("visual_references_text", None)
                manifest.setdefault("world", {}).update(merged)
                if isinstance(visual_references_text, str):
                    manifest.setdefault("world", {})["visual_references"] = (
                        self._split_text_items(visual_references_text)
                    )
                world_refs = manifest.setdefault("world", {}).get("visual_references")
                if isinstance(world_refs, str):
                    manifest["world"]["visual_references"] = self._split_text_items(world_refs)
            self.autofill_service.sync_character_bible(manifest, reason="manual_world")
        elif step_name == "scenes":
            scenes = payload.get("scenes")
            default_scene_type = payload.get("scene_type_default")
            if isinstance(scenes, list):
                manifest["scenes"] = self._normalize_scene_entries(
                    scenes,
                    manifest.get("scenes", []),
                    default_scene_type=default_scene_type,
                )
            else:
                scene_plan_text = str(
                    payload.get("scene_plan_text") or payload.get("scene_plan") or ""
                )
                if scene_plan_text.strip():
                    manifest["scenes"] = self._build_scenes_from_text(
                        manifest,
                        scene_plan_text,
                        default_scene_type=default_scene_type,
                    )
            if manifest.get("scenes"):
                manifest.setdefault("brief", {})["scene_count_hint"] = len(
                    manifest.get("scenes", [])
                )
            for scene in manifest.get("scenes", []):
                if not isinstance(scene, dict):
                    continue
                if scene.get("locked"):
                    scene["state"] = "locked"
                elif scene.get("state") not in self.SCENE_STATES:
                    scene["state"] = "planned"
            manifest.setdefault("approvals", {})
            manifest["approvals"]["script_approved"] = False
            manifest["approvals"]["script_needs_user_approval"] = True
            manifest = self.autofill_service.autofill_step(
                manifest,
                "scenes",
                reason="manual_scenes_sync",
            )
        elif step_name == "script":
            if isinstance(payload.get("scenes"), list):
                self._apply_script_payload(manifest, payload)
            else:
                script_text = str(payload.get("script_text") or "")
                if script_text.strip():
                    self._apply_script_text(manifest, script_text)
            for scene in manifest.get("scenes", []):
                if not isinstance(scene, dict):
                    continue
                if scene.get("locked"):
                    scene["state"] = "locked"
                elif scene.get("state") == "stitched":
                    scene["state"] = "approved"
                elif scene.get("state") not in self.SCENE_STATES:
                    scene["state"] = "planned"
            manifest.setdefault("approvals", {})
            manifest["approvals"]["script_approved"] = False
            manifest["approvals"]["script_needs_user_approval"] = True
        elif step_name == "audio":
            audio_payload = payload.get("audio", payload)
            if isinstance(audio_payload, dict):
                merged = dict(audio_payload)
                sound_effects_text = merged.pop("sound_effects_text", None)
                manifest.setdefault("audio", {}).update(merged)
                if isinstance(sound_effects_text, str):
                    manifest.setdefault("audio", {})["sound_effects"] = self._split_text_items(
                        sound_effects_text
                    )
                sfx = manifest.setdefault("audio", {}).get("sound_effects")
                if isinstance(sfx, str):
                    manifest["audio"]["sound_effects"] = self._split_text_items(sfx)
        elif step_name == "output":
            manifest.setdefault("output", {}).update(payload.get("output", payload))
        elif step_name == "review":
            timeline_payload = payload.get("timeline")
            if isinstance(timeline_payload, dict):
                manifest["timeline"] = timeline_payload
        else:
            raise ValueError(f"Unknown step '{step_name}'")

        manifest = self.validation_service.normalize_manifest(manifest)
        self.save_manifest(project_id, manifest)
        return manifest

    def skip_step(self, project_id: str, step_name: str) -> Dict[str, Any]:
        manifest = self.get_manifest(project_id)
        manifest = self.autofill_service.autofill_step(manifest, step_name)
        manifest = self.validation_service.normalize_manifest(manifest)
        self.save_manifest(project_id, manifest)
        return manifest

    def surprise_step(self, project_id: str, step_name: str) -> Dict[str, Any]:
        manifest = self.get_manifest(project_id)
        manifest = self.autofill_service.surprise_step(manifest, step_name)
        manifest = self.validation_service.normalize_manifest(manifest)
        self.save_manifest(project_id, manifest)
        return manifest

    def lock_scene(self, project_id: str, scene_id: str, locked: bool) -> Dict[str, Any]:
        manifest = self.get_manifest(project_id)
        updated = False
        for scene in manifest.get("scenes", []):
            if scene.get("scene_id") == scene_id:
                scene["locked"] = bool(locked)
                # Locking moves scene into "locked"; unlocking returns to planned unless already generated/approved.
                if locked:
                    scene["state"] = "locked"
                elif scene.get("state") == "locked":
                    scene["state"] = "planned"
                updated = True
                break
        if not updated:
            raise FileNotFoundError(f"Scene '{scene_id}' not found in project '{project_id}'.")
        self.save_manifest(project_id, manifest)
        return manifest

    def completion_state(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        missing_brief = self.validation_service.validate_brief_required(manifest)
        scene_count = len(manifest.get("scenes", []))
        dialogue_count = sum(
            len(scene.get("dialogue", [])) for scene in manifest.get("scenes", [])
        )
        state_counts: Dict[str, int] = {}
        for scene in manifest.get("scenes", []):
            state = str(scene.get("state", "planned"))
            state_counts[state] = state_counts.get(state, 0) + 1
        ready = len(missing_brief) == 0
        script_approved = bool(manifest.get("approvals", {}).get("script_approved"))
        return {
            "ready_for_render": ready and script_approved,
            "missing_required_brief_fields": missing_brief,
            "script_approved": script_approved,
            "script_needs_user_approval": bool(
                manifest.get("approvals", {}).get("script_needs_user_approval")
            ),
            "scene_count": scene_count,
            "dialogue_line_count": dialogue_count,
            "generation_mode": manifest.get("generation_mode", "auto"),
            "scene_state_counts": state_counts,
            "required_brief_fields": REQUIRED_BRIEF_FIELDS,
        }

    def create_job(self, project_id: str) -> Dict[str, Any]:
        manifest = self.get_manifest(project_id)
        job_id = uuid.uuid4().hex[:14]
        now = _utc_now()
        job = {
            "job_id": job_id,
            "project_id": project_id,
            "generation_mode": manifest.get("generation_mode", "auto"),
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "progress": 0.0,
            "warnings": [],
            "errors": [],
            "artifacts": [],
            "current_stage": "queued",
            "current_scene_id": None,
            "current_scene_progress": 0.0,
            "pause_requested": False,
            "cancel_requested": False,
            "control_note": "",
        }
        self._write_json(self._job_path(project_id, job_id), job)
        index = self._read_json(self.jobs_index_path)
        index[job_id] = project_id
        self._write_json(self.jobs_index_path, index)
        return job

    def get_job(self, job_id: str) -> Dict[str, Any]:
        index = self._read_json(self.jobs_index_path)
        project_id = index.get(job_id)
        if not project_id:
            raise FileNotFoundError(f"Job '{job_id}' not found.")
        path = self._job_path(project_id, job_id)
        if not path.exists():
            raise FileNotFoundError(f"Job '{job_id}' metadata is missing.")
        return self._read_json(path)

    def update_job(
        self,
        project_id: str,
        job_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        warnings: Optional[List[Dict[str, Any]]] = None,
        errors: Optional[List[str]] = None,
        artifacts: Optional[List[Dict[str, Any]]] = None,
        append_error: Optional[str] = None,
        current_stage: Optional[str] = None,
        current_scene_id: Optional[str] = None,
        current_scene_progress: Optional[float] = None,
        pause_requested: Optional[bool] = None,
        cancel_requested: Optional[bool] = None,
        control_note: Optional[str] = None,
    ) -> Dict[str, Any]:
        job = self._read_json(self._job_path(project_id, job_id))
        if status:
            job["status"] = status
        if progress is not None:
            job["progress"] = float(progress)
        if warnings is not None:
            job["warnings"] = warnings
        if errors is not None:
            job["errors"] = errors
        if artifacts is not None:
            job["artifacts"] = artifacts
        if append_error:
            job.setdefault("errors", []).append(append_error)
        if current_stage is not None:
            job["current_stage"] = current_stage
        if current_scene_id is not None:
            job["current_scene_id"] = current_scene_id
        if current_scene_progress is not None:
            progress_value = float(current_scene_progress)
            if progress_value < 0:
                progress_value = 0.0
            if progress_value > 1:
                progress_value = 1.0
            job["current_scene_progress"] = progress_value
        if pause_requested is not None:
            job["pause_requested"] = bool(pause_requested)
        if cancel_requested is not None:
            job["cancel_requested"] = bool(cancel_requested)
        if control_note is not None:
            job["control_note"] = str(control_note)
        job["updated_at"] = _utc_now()
        self._write_json(self._job_path(project_id, job_id), job)
        return job

    def request_job_action(self, job_id: str, action: str) -> Dict[str, Any]:
        job = self.get_job(job_id)
        project_id = job["project_id"]
        action = action.lower().strip()
        if action == "pause":
            return self.update_job(
                project_id,
                job_id,
                pause_requested=True,
                control_note="Pause requested by user.",
            )
        if action == "resume":
            current_stage = str(job.get("current_stage") or "").strip().lower()
            current_scene_id = str(job.get("current_scene_id") or "").strip()
            resume_note = "Resume requested by user."
            if current_stage == "awaiting_scene_approval" and current_scene_id:
                manifest = self.get_manifest(project_id)
                scene = self._find_scene(manifest, current_scene_id)
                scene_state = str(scene.get("state", "planned")) if scene else ""
                if scene_state == "generated":
                    self.set_scene_approval(
                        project_id,
                        current_scene_id,
                        True,
                        note="Approved by resume control.",
                    )
                    resume_note = (
                        f"Resume requested by user. Approved {current_scene_id} and continuing."
                    )
            return self.update_job(
                project_id,
                job_id,
                pause_requested=False,
                status="running",
                control_note=resume_note,
            )
        if action == "cancel":
            return self.update_job(
                project_id,
                job_id,
                cancel_requested=True,
                control_note="Cancel requested by user.",
            )
        raise ValueError(f"Unknown job action '{action}'.")

    def set_script_approval(
        self,
        project_id: str,
        approved: bool,
        reviewer_note: str = "",
    ) -> Dict[str, Any]:
        manifest = self.get_manifest(project_id)
        approvals = manifest.setdefault("approvals", {})
        approvals["script_approved"] = bool(approved)
        approvals["script_needs_user_approval"] = not bool(approved)
        approvals["script_last_reviewed_at"] = _utc_now()
        if reviewer_note:
            approvals["script_reviewer_note"] = reviewer_note
        self.save_manifest(project_id, manifest)
        return manifest

    def set_generation_mode(self, project_id: str, mode: str) -> Dict[str, Any]:
        normalized = str(mode or "").strip().lower()
        if normalized not in ("auto", "directed"):
            raise ValueError("generation_mode must be either 'auto' or 'directed'.")
        manifest = self.get_manifest(project_id)
        manifest["generation_mode"] = normalized
        self.save_manifest(project_id, manifest)
        return manifest

    def set_scene_state(
        self,
        project_id: str,
        scene_id: str,
        state: str,
        *,
        note: str = "",
        force: bool = False,
    ) -> Dict[str, Any]:
        target_state = str(state or "").strip().lower()
        if target_state not in self.SCENE_STATES:
            raise ValueError(
                "scene state must be one of: planned, generated, approved, locked, stitched."
            )
        manifest = self.get_manifest(project_id)
        scene = self._find_scene(manifest, scene_id)
        if scene is None:
            raise FileNotFoundError(f"Scene '{scene_id}' not found in project '{project_id}'.")

        current_state = str(scene.get("state", "planned"))
        if not force and target_state != current_state:
            allowed = self.SCENE_TRANSITIONS.get(current_state, set())
            if target_state not in allowed:
                raise ValueError(
                    f"Invalid scene state transition: {current_state} -> {target_state}"
                )

        scene["state"] = target_state
        scene["locked"] = target_state == "locked"
        state_history = scene.setdefault("state_history", [])
        state_history.append(
            {
                "from_state": current_state,
                "to_state": target_state,
                "timestamp": _utc_now(),
                "note": note or "",
            }
        )

        self.save_manifest(project_id, manifest)
        return manifest

    def set_scene_approval(
        self,
        project_id: str,
        scene_id: str,
        approved: bool,
        *,
        note: str = "",
    ) -> Dict[str, Any]:
        manifest = self.get_manifest(project_id)
        scene = self._find_scene(manifest, scene_id)
        if scene is None:
            raise FileNotFoundError(f"Scene '{scene_id}' not found in project '{project_id}'.")

        current_state = str(scene.get("state", "planned"))
        if approved:
            target_state = "approved"
        else:
            target_state = "approved" if current_state == "stitched" else "planned"

        return self.set_scene_state(
            project_id,
            scene_id,
            target_state,
            note=note or ("Approved by user." if approved else "Approval removed by user."),
            force=True,
        )

    def update_scene_trim(
        self,
        project_id: str,
        scene_id: str,
        *,
        start_trim: float,
        end_trim: float,
    ) -> Dict[str, Any]:
        if start_trim < 0 or end_trim < 0:
            raise ValueError("Trim values must be non-negative.")
        manifest = self.get_manifest(project_id)
        scene = self._find_scene(manifest, scene_id)
        if scene is None:
            raise FileNotFoundError(f"Scene '{scene_id}' not found in project '{project_id}'.")

        timeline = manifest.setdefault("timeline", {})
        trims = timeline.setdefault("scene_trims", {})
        trims[scene_id] = {
            "start_trim": round(float(start_trim), 3),
            "end_trim": round(float(end_trim), 3),
        }
        self.save_manifest(project_id, manifest)
        return manifest

    def update_subtitle_line(
        self,
        project_id: str,
        scene_id: str,
        line_index: int,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        manifest = self.get_manifest(project_id)
        scene = self._find_scene(manifest, scene_id)
        if scene is None:
            raise FileNotFoundError(f"Scene '{scene_id}' not found in project '{project_id}'.")

        dialogue = scene.get("dialogue", [])
        if line_index < 0 or line_index >= len(dialogue):
            raise IndexError(
                f"line_index '{line_index}' is out of range for scene '{scene_id}'."
            )

        line = dialogue[line_index]
        for field in ["line", "emotion", "pause_notes", "speaker_id"]:
            if field in payload:
                line[field] = payload.get(field)

        if "start_second" in payload:
            line["start_second"] = float(payload.get("start_second") or 0.0)
        if "end_second" in payload:
            line["end_second"] = float(payload.get("end_second") or 0.0)

        start_second = float(line.get("start_second", 0.0))
        end_second = float(line.get("end_second", 0.0))
        if end_second < start_second:
            raise ValueError("end_second must be greater than or equal to start_second.")

        self._rebuild_timeline_entries(manifest)
        self.save_manifest(project_id, manifest)
        return manifest

    def update_scene_content(
        self,
        project_id: str,
        scene_id: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        manifest = self.get_manifest(project_id)
        scenes = manifest.get("scenes", [])
        target_index = -1
        target_scene: Dict[str, Any] = {}
        for idx, scene in enumerate(scenes):
            if isinstance(scene, dict) and scene.get("scene_id") == scene_id:
                target_index = idx
                target_scene = scene
                break
        if target_index < 0:
            raise FileNotFoundError(f"Scene '{scene_id}' not found in project '{project_id}'.")

        updates = dict(payload or {})
        if "events_text" in updates and "events" not in updates:
            updates["events"] = self._split_text_items(str(updates.get("events_text") or ""))
        if "characters_text" in updates and "characters_in_scene" not in updates:
            updates["characters_in_scene"] = self._split_text_items(
                str(updates.get("characters_text") or "")
            )
        if "reference_images_text" in updates and "reference_images" not in updates:
            updates["reference_images"] = self._split_text_items(
                str(updates.get("reference_images_text") or "")
            )

        allowed_fields = {
            "scene_id",
            "scene_prompt",
            "conversation",
            "scene_goal",
            "scene_type",
            "scene_description",
            "spoken_direction",
            "image_usage_instructions",
            "location_time",
            "characters_in_scene",
            "events",
            "dialogue",
            "narration",
            "shot_list",
            "reference_images",
            "continuity",
            "prompt_memory",
            "builder",
        }
        filtered: Dict[str, Any] = {}
        for key, value in updates.items():
            if key in allowed_fields:
                filtered[key] = value

        merged = copy.deepcopy(target_scene)
        merged.update(filtered)
        merged["scene_id"] = scene_id

        normalized_scenes = self._normalize_scene_entries(
            [merged],
            scenes,
            default_scene_type=merged.get("scene_type"),
        )
        if not normalized_scenes:
            raise ValueError("Scene update produced invalid scene payload.")
        normalized_scene = normalized_scenes[0]

        # Mark scene as needing regeneration and clear stale debug output.
        if normalized_scene.get("locked"):
            normalized_scene["state"] = "locked"
        else:
            normalized_scene["state"] = "planned"
        normalized_scene.pop("render_debug", None)

        scenes[target_index] = normalized_scene
        manifest["scenes"] = scenes
        manifest = self.validation_service.normalize_manifest(manifest)

        approvals = manifest.setdefault("approvals", {})
        approvals["script_approved"] = False
        approvals["script_needs_user_approval"] = True

        self.save_manifest(project_id, manifest)
        return manifest

    def estimate_project(self, project_id: str) -> Dict[str, Any]:
        manifest = self.get_manifest(project_id)
        scenes = manifest.get("scenes", [])
        dialogue_lines = sum(len(scene.get("dialogue", [])) for scene in scenes)

        total_scene_seconds = 0.0
        for scene in scenes:
            total_scene_seconds += self._scene_duration_seconds(scene)

        scene_count = len(scenes)
        mode = manifest.get("generation_mode", "auto")
        base_overhead_minutes = 1.3 if mode == "auto" else 1.8
        generation_minutes = base_overhead_minutes + (scene_count * 0.9) + (
            total_scene_seconds / 45.0
        )

        video_cost = scene_count * 0.18
        voice_cost = dialogue_lines * 0.015
        resolution_factor = 1.0
        resolution = str(manifest.get("output", {}).get("resolution", "1080p")).lower()
        if "4k" in resolution:
            resolution_factor = 1.8
        elif "720" in resolution:
            resolution_factor = 0.8

        estimated_cost = round((video_cost + voice_cost) * resolution_factor, 2)

        return {
            "project_id": project_id,
            "generation_mode": mode,
            "scene_count": scene_count,
            "dialogue_line_count": dialogue_lines,
            "estimated_video_seconds": round(total_scene_seconds, 2),
            "estimated_render_minutes": round(generation_minutes, 2),
            "estimated_api_cost_usd": estimated_cost,
            "assumptions": {
                "video_cost_per_scene_usd": 0.18,
                "voice_cost_per_line_usd": 0.015,
                "resolution_factor": resolution_factor,
            },
        }

    def create_snapshot(self, project_id: str, label: str = "manual") -> Dict[str, Any]:
        manifest = self.get_manifest(project_id)
        snapshot_id = f"snap_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"
        payload = {
            "snapshot_id": snapshot_id,
            "project_id": project_id,
            "label": (label or "manual").strip() or "manual",
            "created_at": _utc_now(),
            "manifest": manifest,
        }
        self._write_json(self._snapshot_path(project_id, snapshot_id), payload)
        return {
            "snapshot_id": snapshot_id,
            "project_id": project_id,
            "label": payload["label"],
            "created_at": payload["created_at"],
        }

    def list_snapshots(self, project_id: str) -> List[Dict[str, Any]]:
        project_dir = self._project_dir(project_id)
        if not project_dir.exists():
            raise FileNotFoundError(f"Project '{project_id}' not found.")
        snapshot_dir = self._snapshot_dir(project_id)
        if not snapshot_dir.exists():
            return []

        snapshots: List[Dict[str, Any]] = []
        for path in sorted(snapshot_dir.glob("*.json")):
            payload = self._read_json(path)
            snapshots.append(
                {
                    "snapshot_id": payload.get("snapshot_id", path.stem),
                    "project_id": payload.get("project_id", project_id),
                    "label": payload.get("label", "snapshot"),
                    "created_at": payload.get("created_at", ""),
                    "path": str(path.relative_to(self._project_dir(project_id))).replace(
                        "\\", "/"
                    ),
                }
            )
        return snapshots

    def rollback_snapshot(self, project_id: str, snapshot_id: str) -> Dict[str, Any]:
        path = self._snapshot_path(project_id, snapshot_id)
        if not path.exists():
            raise FileNotFoundError(f"Snapshot '{snapshot_id}' not found for '{project_id}'.")
        payload = self._read_json(path)
        manifest = payload.get("manifest")
        if not isinstance(manifest, dict):
            raise ValueError(f"Snapshot '{snapshot_id}' has invalid manifest payload.")

        manifest["project_id"] = project_id
        manifest["updated_at"] = _utc_now()
        manifest = self.validation_service.normalize_manifest(manifest)
        self.save_manifest(project_id, manifest)
        return manifest

    def create_export_bundle(self, project_id: str) -> Dict[str, str]:
        project_dir = self._project_dir(project_id)
        if not project_dir.exists():
            raise FileNotFoundError(f"Project '{project_id}' not found.")
        exports_dir = self._export_dir(project_id)
        exports_dir.mkdir(parents=True, exist_ok=True)
        bundle_name = (
            f"{project_id}_bundle_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.zip"
        )
        bundle_path = exports_dir / bundle_name

        include_files = []
        manifest_file = project_dir / "manifest.json"
        autofill_file = project_dir / "autofill_log.json"
        if manifest_file.exists():
            include_files.append(manifest_file)
        if autofill_file.exists():
            include_files.append(autofill_file)

        for relative_root in [
            "assets/video",
            "assets/audio",
            "assets/subtitles",
            "reports",
            "jobs",
        ]:
            folder = project_dir / relative_root
            if not folder.exists():
                continue
            include_files.extend([path for path in folder.rglob("*") if path.is_file()])

        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_path in sorted(include_files):
                arcname = str(file_path.relative_to(project_dir)).replace("\\", "/")
                archive.write(file_path, arcname=arcname)

        relative = str(bundle_path.relative_to(project_dir)).replace("\\", "/")
        return {
            "type": "export_bundle",
            "path": relative,
            "url": f"/api/v2/projects/{project_id}/files/{relative}",
        }

    def update_character_bible(
        self, project_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        manifest = self.get_manifest(project_id)
        bible = manifest.setdefault("character_bible", {})
        for key, value in payload.items():
            if not isinstance(value, dict):
                continue
            current = bible.setdefault(key, {})
            current.update(value)
        self.save_manifest(project_id, manifest)
        return manifest

    def attach_character_reference_image(
        self,
        project_id: str,
        character_id: str,
        relative_path: str,
    ) -> Dict[str, Any]:
        manifest = self.get_manifest(project_id)
        target_character = None
        for character in manifest.get("characters", []):
            if character.get("character_id") == character_id:
                target_character = character
                break
        if target_character is None:
            raise FileNotFoundError(
                f"Character '{character_id}' not found in project '{project_id}'."
            )

        relative_clean = str(relative_path).replace("\\", "/")
        target_refs = target_character.setdefault("reference_images", [])
        if relative_clean not in target_refs:
            target_refs.append(relative_clean)

        bible = manifest.setdefault("character_bible", {}).setdefault(character_id, {})
        bible_refs = bible.setdefault("reference_images", [])
        if relative_clean not in bible_refs:
            bible_refs.append(relative_clean)

        self.save_manifest(project_id, manifest)
        return manifest

    def update_timeline(self, project_id: str, timeline: Dict[str, Any]) -> Dict[str, Any]:
        manifest = self.get_manifest(project_id)
        manifest["timeline"] = timeline
        manifest = self.validation_service.normalize_manifest(manifest)
        self.save_manifest(project_id, manifest)
        return manifest

    def regenerate_scene(
        self,
        project_id: str,
        scene_id: str,
        *,
        branch: bool = False,
        activate: bool = True,
    ) -> Dict[str, Any]:
        manifest = self.get_manifest(project_id)
        variants = manifest.setdefault("scene_variants", {}).setdefault(scene_id, [])
        branch_index = len(variants) + 1
        original_scene = None
        original_index = -1
        for idx, scene in enumerate(manifest.get("scenes", [])):
            if scene.get("scene_id") == scene_id:
                original_scene = copy.deepcopy(scene)
                original_index = idx
                break
        if original_scene is None:
            raise FileNotFoundError(f"Scene '{scene_id}' not found in project '{project_id}'.")

        scene_variant = self.autofill_service.regenerate_scene_content(
            manifest,
            scene_id=scene_id,
            branch_index=branch_index,
        )

        variant_id = f"{scene_id}_v{branch_index:02d}"
        variants.append(
            {
                "variant_id": variant_id,
                "created_at": _utc_now(),
                "scene": copy.deepcopy(scene_variant),
            }
        )

        if branch and not activate:
            manifest["scenes"][original_index] = original_scene
        else:
            for scene in manifest.get("scenes", []):
                if scene.get("scene_id") == scene_id:
                    scene["active_variant_id"] = variant_id
                    scene["state"] = "planned"
                    scene["locked"] = False
                    break

        manifest = self.autofill_service.autofill_step(
            manifest, "script", reason="timeline_resync_after_regen"
        )

        manifest.setdefault("approvals", {})
        manifest["approvals"]["script_approved"] = False
        manifest["approvals"]["script_needs_user_approval"] = True
        self.save_manifest(project_id, manifest)
        return manifest

    def activate_scene_variant(
        self, project_id: str, scene_id: str, variant_id: str
    ) -> Dict[str, Any]:
        manifest = self.get_manifest(project_id)
        variants = manifest.get("scene_variants", {}).get(scene_id, [])
        selected = None
        for variant in variants:
            if variant.get("variant_id") == variant_id:
                selected = copy.deepcopy(variant.get("scene"))
                break
        if not selected:
            raise FileNotFoundError(f"Variant '{variant_id}' not found for scene '{scene_id}'.")

        replaced = False
        for idx, scene in enumerate(manifest.get("scenes", [])):
            if scene.get("scene_id") == scene_id:
                selected["scene_id"] = scene_id
                selected["active_variant_id"] = variant_id
                selected["state"] = "planned"
                selected["locked"] = False
                manifest["scenes"][idx] = selected
                replaced = True
                break
        if not replaced:
            raise FileNotFoundError(f"Scene '{scene_id}' not found.")

        manifest = self.autofill_service.autofill_step(
            manifest, "script", reason="timeline_resync_after_variant_activate"
        )
        manifest.setdefault("approvals", {})
        manifest["approvals"]["script_approved"] = False
        manifest["approvals"]["script_needs_user_approval"] = True
        self.save_manifest(project_id, manifest)
        return manifest

    def list_artifacts(self, project_id: str) -> List[Dict[str, str]]:
        project_dir = self._project_dir(project_id)
        if not project_dir.exists():
            raise FileNotFoundError(f"Project '{project_id}' not found.")

        artifacts: List[Dict[str, str]] = []
        for relative in [
            "manifest.json",
            "autofill_log.json",
        ]:
            path = project_dir / relative
            if path.exists():
                artifacts.append(
                    {
                        "type": "metadata",
                        "path": relative.replace("\\", "/"),
                        "url": f"/api/v2/projects/{project_id}/files/{relative}",
                    }
                )

        for base_folder, artifact_type in [
            ("assets/video", "video"),
            ("assets/audio", "audio"),
            ("assets/subtitles", "subtitles"),
            ("assets/references", "reference"),
            ("reports", "report"),
            ("jobs", "job"),
            ("exports", "export"),
            ("snapshots", "snapshot"),
        ]:
            folder = project_dir / base_folder
            if not folder.exists():
                continue
            for file_path in sorted(folder.rglob("*")):
                if not file_path.is_file():
                    continue
                relative = str(file_path.relative_to(project_dir)).replace("\\", "/")
                artifacts.append(
                    {
                        "type": artifact_type,
                        "path": relative,
                        "url": f"/api/v2/projects/{project_id}/files/{relative}",
                    }
                )
        return artifacts

    def get_project_file(self, project_id: str, relative_path: str) -> Tuple[str, str]:
        project_dir = self._project_dir(project_id).resolve()
        candidate = (project_dir / relative_path).resolve()
        if not str(candidate).startswith(str(project_dir)):
            raise PermissionError("Invalid path.")
        if not candidate.exists():
            raise FileNotFoundError(relative_path)
        return str(candidate.parent), candidate.name

    def project_paths(self, project_id: str) -> Dict[str, str]:
        self._ensure_project_layout(project_id)
        project_dir = self._project_dir(project_id)
        return {
            "project_dir": str(project_dir),
            "manifest": str(project_dir / "manifest.json"),
            "autofill_log": str(project_dir / "autofill_log.json"),
            "assets_video_dir": str(project_dir / "assets" / "video"),
            "assets_audio_dir": str(project_dir / "assets" / "audio"),
            "assets_subtitles_dir": str(project_dir / "assets" / "subtitles"),
            "assets_reference_dir": str(project_dir / "assets" / "references"),
            "reports_dir": str(project_dir / "reports"),
            "jobs_dir": str(project_dir / "jobs"),
            "snapshots_dir": str(project_dir / "snapshots"),
            "exports_dir": str(project_dir / "exports"),
        }

    def _apply_script_payload(self, manifest: Dict[str, Any], payload: Dict[str, Any]) -> None:
        scenes_payload = payload.get("scenes")
        if isinstance(scenes_payload, list):
            scene_map = {scene.get("scene_id"): scene for scene in manifest.get("scenes", [])}
            for scene in scenes_payload:
                scene_id = scene.get("scene_id")
                if not scene_id:
                    continue
                existing = scene_map.get(scene_id)
                if not existing:
                    manifest.setdefault("scenes", []).append(scene)
                else:
                    if "dialogue" in scene:
                        existing["dialogue"] = scene["dialogue"]
                    if "shot_list" in scene:
                        existing["shot_list"] = scene["shot_list"]
                    if "events" in scene:
                        existing["events"] = scene["events"]

    def _apply_description_to_manifest(
        self,
        manifest: Dict[str, Any],
        description: str,
        *,
        reason: str,
    ) -> None:
        text = str(description or "").strip()
        if not text:
            return

        brief = manifest.setdefault("brief", {})
        brief["description"] = text
        self._log_description_field(manifest, "brief.description", text, reason)

        if brief.get("scene_count_hint") in (None, "", 0):
            hint = self._infer_scene_count_hint(text)
            if hint:
                brief["scene_count_hint"] = hint
                self._log_description_field(
                    manifest,
                    "brief.scene_count_hint",
                    hint,
                    reason,
                )

        if not manifest.get("characters"):
            primary_name = self._extract_primary_character_name(text)
            if primary_name:
                manifest["characters"] = self._normalize_character_entries(
                    [
                        {
                            "name": primary_name,
                            "role": "protagonist",
                            "appearance": "Age 20-35, natural styling",
                            "personality": "calm and relatable",
                            "constraints": "Avoid resemblance to real people.",
                            "voice": {
                                "voice_provider": "native",
                                "voice_id": "",
                                "speaking_style": "conversational, warm, reassuring",
                            },
                        }
                    ]
                )
                self._log_description_field(
                    manifest,
                    "characters",
                    manifest["characters"],
                    reason,
                )

        if not manifest.get("scenes"):
            extracted_scene_plan = self._extract_scene_plan_from_description(text)
            if extracted_scene_plan:
                manifest["scenes"] = self._build_scenes_from_plan(
                    manifest,
                    extracted_scene_plan,
                )
                self._log_description_field(
                    manifest,
                    "scenes",
                    manifest["scenes"],
                    reason,
                )

        planner_patch = self._planner_patch_from_description(manifest, text)
        if planner_patch:
            self._merge_description_plan(manifest, planner_patch, reason)

        lower = text.lower()
        inferred = {
            "video_type": self._infer_video_type(lower),
            "platform": self._infer_platform(lower),
            "tone": self._infer_tone(lower),
            "duration_seconds": self._infer_duration_seconds(lower),
            "language": self._infer_language(lower),
            "accent": self._infer_accent(lower),
            "target_audience": self._infer_target_audience(text),
            "goal": self._infer_goal(lower, text),
        }

        platform = inferred.get("platform")
        inferred["aspect_ratio"] = self._infer_aspect_ratio(lower, platform)

        for field, value in inferred.items():
            if value in (None, "", []):
                continue
            if brief.get(field) in (None, "", []):
                brief[field] = value
                self._log_description_field(
                    manifest,
                    f"brief.{field}",
                    value,
                    reason,
                )

    def _planner_patch_from_description(
        self,
        manifest: Dict[str, Any],
        description: str,
    ) -> Dict[str, Any]:
        planner = self.description_planner_service
        if planner is None:
            return {}
        try:
            return planner.plan_manifest_patch(
                description,
                context={
                    "brief": copy.deepcopy(manifest.get("brief", {})),
                    "characters": copy.deepcopy(manifest.get("characters", [])),
                    "world": copy.deepcopy(manifest.get("world", {})),
                    "scenes": copy.deepcopy(manifest.get("scenes", [])),
                },
            )
        except Exception:
            return {}

    def _merge_description_plan(
        self,
        manifest: Dict[str, Any],
        planner_patch: Dict[str, Any],
        reason: str,
    ) -> None:
        brief = manifest.setdefault("brief", {})
        brief_patch = planner_patch.get("brief")
        if isinstance(brief_patch, dict):
            for field, value in brief_patch.items():
                if value in (None, "", []):
                    continue
                normalized_value = value
                if field in {"duration_seconds", "scene_count_hint"}:
                    try:
                        normalized_value = int(value)
                    except (TypeError, ValueError):
                        continue
                if brief.get(field) in (None, "", []):
                    brief[field] = normalized_value
                    self._log_description_field(
                        manifest,
                        f"brief.{field}",
                        normalized_value,
                        reason,
                        source="gemini_inference",
                        seed_fragment="description_planner",
                    )

        characters_patch = planner_patch.get("characters")
        if isinstance(characters_patch, list) and not manifest.get("characters"):
            normalized_characters = self._normalize_character_entries(characters_patch)
            if normalized_characters:
                manifest["characters"] = normalized_characters
                self._log_description_field(
                    manifest,
                    "characters",
                    normalized_characters,
                    reason,
                    source="gemini_inference",
                    seed_fragment="description_planner",
                )

        world_patch = planner_patch.get("world")
        if isinstance(world_patch, dict):
            world = manifest.setdefault("world", {})
            for field, value in world_patch.items():
                if value in (None, "", []):
                    continue
                if world.get(field) in (None, "", []):
                    world[field] = value
                    self._log_description_field(
                        manifest,
                        f"world.{field}",
                        value,
                        reason,
                        source="gemini_inference",
                        seed_fragment="description_planner",
                    )

        planner_scene_plan = planner_patch.get("scene_plan")
        if isinstance(planner_scene_plan, list) and not manifest.get("scenes"):
            manifest["scenes"] = self._build_scenes_from_plan(
                manifest,
                planner_scene_plan,
            )
            brief = manifest.setdefault("brief", {})
            if brief.get("scene_count_hint") in (None, "", 0):
                brief["scene_count_hint"] = len(manifest["scenes"])
            self._log_description_field(
                manifest,
                "scenes",
                manifest["scenes"],
                reason,
                source="gemini_inference",
                seed_fragment="description_planner",
            )

        scene_plan_text = planner_patch.get("scene_plan_text")
        if (
            isinstance(scene_plan_text, str)
            and scene_plan_text.strip()
            and not manifest.get("scenes")
        ):
            manifest["scenes"] = self._build_scenes_from_text(manifest, scene_plan_text)
            self._log_description_field(
                manifest,
                "scenes",
                manifest["scenes"],
                reason,
                source="gemini_inference",
                seed_fragment="description_planner",
            )

        script_text = planner_patch.get("script_text")
        has_existing_dialogue = any(
            isinstance(scene, dict) and scene.get("dialogue")
            for scene in manifest.get("scenes", [])
        )
        if (
            isinstance(script_text, str)
            and script_text.strip()
            and manifest.get("scenes")
            and not has_existing_dialogue
        ):
            self._apply_script_text(manifest, script_text)
            self._log_description_field(
                manifest,
                "script_text",
                script_text.strip(),
                reason,
                source="gemini_inference",
                seed_fragment="description_planner",
            )

        audio_patch = planner_patch.get("audio")
        if isinstance(audio_patch, dict):
            audio = manifest.setdefault("audio", {})
            for field, value in audio_patch.items():
                if value in (None, "", []):
                    continue
                if audio.get(field) in (None, "", []):
                    audio[field] = value
                    self._log_description_field(
                        manifest,
                        f"audio.{field}",
                        value,
                        reason,
                        source="gemini_inference",
                        seed_fragment="description_planner",
                    )

        output_patch = planner_patch.get("output")
        if isinstance(output_patch, dict):
            output = manifest.setdefault("output", {})
            for field, value in output_patch.items():
                if value in (None, "", []):
                    continue
                normalized_value = value
                if field == "fps":
                    try:
                        normalized_value = int(value)
                    except (TypeError, ValueError):
                        continue
                if output.get(field) in (None, "", []):
                    output[field] = normalized_value
                    self._log_description_field(
                        manifest,
                        f"output.{field}",
                        normalized_value,
                        reason,
                        source="gemini_inference",
                        seed_fragment="description_planner",
                    )

        assumptions = planner_patch.get("assumptions")
        if isinstance(assumptions, list):
            cleaned = []
            for item in assumptions:
                text = str(item).strip()
                if text:
                    cleaned.append(text)
            if cleaned:
                bucket = manifest.setdefault("planner_assumptions", [])
                for note in cleaned:
                    if note not in bucket:
                        bucket.append(note)
                        self._log_description_field(
                            manifest,
                            "planner_assumptions",
                            note,
                            reason,
                            source="gemini_inference",
                            seed_fragment="description_planner",
                        )

    @staticmethod
    def _infer_video_type(lower_text: str) -> str:
        rules = [
            ("testimonial", "testimonial"),
            ("tutorial", "tutorial"),
            ("how to", "tutorial"),
            ("explainer", "explainer"),
            ("social skit", "social skit"),
            ("skit", "social skit"),
            ("story", "story"),
            ("ad", "ad"),
            ("advert", "ad"),
        ]
        for keyword, value in rules:
            if keyword in lower_text:
                return value
        return "ad"

    @staticmethod
    def _infer_platform(lower_text: str) -> str:
        rules = [
            ("tiktok", "TikTok"),
            ("reels", "Instagram"),
            ("instagram", "Instagram"),
            ("youtube shorts", "YouTube"),
            ("youtube", "YouTube"),
            ("linkedin", "LinkedIn"),
            ("website", "Website"),
        ]
        for keyword, value in rules:
            if keyword in lower_text:
                return value
        return "TikTok"

    @staticmethod
    def _infer_tone(lower_text: str) -> str:
        rules = [
            ("funny", "funny"),
            ("humor", "funny"),
            ("comedic", "funny"),
            ("serious", "serious"),
            ("emotional", "emotional"),
            ("luxury", "luxury"),
            ("gritty", "gritty"),
        ]
        for keyword, value in rules:
            if keyword in lower_text:
                return value
        return "engaging"

    @staticmethod
    def _infer_duration_seconds(lower_text: str) -> Optional[int]:
        match = re.search(r"\b(\d{1,3})\s*(?:s|sec|secs|second|seconds)\b", lower_text)
        if match:
            try:
                value = int(match.group(1))
                return max(6, min(180, value))
            except Exception:
                return None
        if "8 second" in lower_text or "8s" in lower_text:
            return 8
        if "short" in lower_text:
            return 15
        return None

    @staticmethod
    def _infer_language(lower_text: str) -> str:
        mapping = {
            "english": "English",
            "spanish": "Spanish",
            "french": "French",
            "german": "German",
            "arabic": "Arabic",
            "portuguese": "Portuguese",
            "hindi": "Hindi",
        }
        for keyword, value in mapping.items():
            if keyword in lower_text:
                return value
        return "English"

    @staticmethod
    def _infer_accent(lower_text: str) -> str:
        rules = [
            ("american accent", "American"),
            ("british accent", "British"),
            ("nigerian accent", "Nigerian"),
            ("indian accent", "Indian"),
            ("australian accent", "Australian"),
            ("neutral accent", "neutral"),
        ]
        for keyword, value in rules:
            if keyword in lower_text:
                return value
        return "neutral American"

    @staticmethod
    def _infer_target_audience(description: str) -> str:
        text = str(description)
        match = re.search(
            r"\bfor\s+([a-zA-Z0-9 ,\-]{3,100}?)(?:[.:\n]|$)",
            text,
            flags=re.I,
        )
        if match:
            value = match.group(1).strip(" ,.-")
            if value:
                return value
        return "general audience"

    @staticmethod
    def _infer_goal(lower_text: str, text: str) -> str:
        rules = [
            (("sign up", "signup", "register"), "Drive signups."),
            (("book a demo", "book demo"), "Drive demo bookings."),
            (("buy", "purchase", "sales"), "Drive purchases."),
            (("download", "install"), "Drive app installs."),
            (("follow", "subscribe"), "Increase follows/subscribers."),
            (("awareness",), "Increase brand awareness."),
        ]
        for keywords, goal in rules:
            for keyword in keywords:
                if " " in keyword:
                    found = keyword in lower_text
                else:
                    found = bool(
                        re.search(
                            rf"\b{re.escape(keyword)}\b",
                            lower_text,
                        )
                    )
                if found:
                    return goal

        sentence_match = re.split(r"[.!?]\s*", str(text).strip(), maxsplit=1)
        first_sentence = sentence_match[0].strip() if sentence_match else ""
        if first_sentence:
            return f"Deliver this message clearly: {first_sentence[:120]}"
        return "Make viewers curious and ready to act."

    @staticmethod
    def _infer_aspect_ratio(lower_text: str, platform: str) -> str:
        explicit = re.search(r"\b(16:9|9:16|1:1)\b", lower_text)
        if explicit:
            return explicit.group(1)
        platform_lower = str(platform or "").lower()
        if "tiktok" in platform_lower or "instagram" in platform_lower:
            return "9:16"
        if "youtube" in platform_lower:
            return "16:9"
        return "9:16"

    @staticmethod
    def _log_description_field(
        manifest: Dict[str, Any],
        field_path: str,
        value: Any,
        reason: str,
        source: str = "description_inference",
        seed_fragment: str = "description",
    ) -> None:
        manifest.setdefault("autofill_log", []).append(
            {
                "field_path": field_path,
                "value": value,
                "source": source,
                "reason": reason,
                "generator_version": "v2.0",
                "seed_fragment": seed_fragment,
                "timestamp": utc_now_iso(),
            }
        )

    def _apply_script_text(self, manifest: Dict[str, Any], script_text: str) -> None:
        scenes = manifest.get("scenes", [])
        if not scenes:
            scenes = self._build_scenes_from_text(manifest, "Intro\nCore message\nCTA")
            manifest["scenes"] = scenes
        for scene in scenes:
            if isinstance(scene, dict):
                scene["dialogue"] = []

        scene_lookup = {
            scene.get("scene_id"): scene
            for scene in scenes
            if isinstance(scene, dict) and scene.get("scene_id")
        }
        current_scene = scenes[0]
        line_cursor = 0
        for raw_line in str(script_text).splitlines():
            line = raw_line.strip()
            if not line:
                continue
            header_match = re.match(r"^(scene[_\s-]*\d+|scene\s+\d+)\s*[:\-]?\s*(.*)$", line, re.I)
            if header_match:
                scene_ref = header_match.group(1).lower().replace(" ", "_").replace("-", "_")
                scene_ref = scene_ref.replace("__", "_")
                normalized_scene_id = scene_ref if scene_ref.startswith("scene_") else scene_ref.replace("scene", "scene_")
                if normalized_scene_id in scene_lookup:
                    current_scene = scene_lookup[normalized_scene_id]
                else:
                    goal = header_match.group(2).strip() or f"Scene {normalized_scene_id}"
                    new_scene = {
                        "scene_id": normalized_scene_id,
                        "scene_goal": goal,
                        "scene_type": self._infer_scene_type_from_goal(goal),
                        "location_time": manifest.get("world", {}).get(
                            "setting", "auto-generated setting"
                        ),
                        "characters_in_scene": [
                            item.get("character_id")
                            for item in manifest.get("characters", [])
                            if isinstance(item, dict) and item.get("character_id")
                        ][:2],
                        "events": [goal],
                        "dialogue": [],
                        "narration": "",
                        "shot_list": [],
                        "locked": False,
                        "state": "planned",
                    }
                    scenes.append(new_scene)
                    scene_lookup[normalized_scene_id] = new_scene
                    current_scene = new_scene
                if header_match.group(2).strip():
                    current_scene["scene_goal"] = header_match.group(2).strip()
                continue

            speaker = "narrator"
            text_value = line
            if ":" in line:
                maybe_speaker, maybe_text = line.split(":", 1)
                if maybe_text.strip():
                    speaker = maybe_speaker.strip().lower().replace(" ", "_")
                    text_value = maybe_text.strip()
            start_second = round(line_cursor * 2.8, 2)
            end_second = round(start_second + 2.6, 2)
            current_scene.setdefault("dialogue", []).append(
                {
                    "speaker_id": speaker or "narrator",
                    "line": text_value,
                    "emotion": "natural",
                    "pause_notes": "",
                    "start_second": start_second,
                    "end_second": end_second,
                }
            )
            line_cursor += 1

        self._rebuild_timeline_entries(manifest)

    def _build_scenes_from_text(
        self,
        manifest: Dict[str, Any],
        text: str,
        *,
        default_scene_type: Any = None,
    ) -> List[Dict[str, Any]]:
        lines = self._split_text_items(text)
        if not lines:
            return manifest.get("scenes", [])

        existing_by_id = {
            scene.get("scene_id"): scene
            for scene in manifest.get("scenes", [])
            if isinstance(scene, dict) and scene.get("scene_id")
        }
        world_setting = manifest.get("world", {}).get("setting", "auto-generated setting")
        characters = [
            item.get("character_id")
            for item in manifest.get("characters", [])
            if isinstance(item, dict) and item.get("character_id")
        ]
        generated = []
        for index, goal in enumerate(lines, start=1):
            scene_id = f"scene_{index:03d}"
            if scene_id in existing_by_id and existing_by_id[scene_id].get("locked"):
                generated.append(existing_by_id[scene_id])
                continue

            existing = existing_by_id.get(scene_id, {})
            scene_type = self._normalize_scene_type(
                existing.get("scene_type")
                or default_scene_type
                or self._infer_scene_type_from_goal(goal)
            )
            generated.append(
                {
                    "scene_id": scene_id,
                    "scene_prompt": goal,
                    "conversation": existing.get("conversation", []),
                    "scene_goal": goal,
                    "scene_type": scene_type,
                    "location_time": existing.get("location_time", world_setting),
                    "characters_in_scene": existing.get(
                        "characters_in_scene",
                        characters[:2],
                    ),
                    "events": existing.get("events", [goal]),
                    "dialogue": existing.get("dialogue", []),
                    "narration": existing.get("narration", ""),
                    "shot_list": existing.get("shot_list", []),
                    "locked": bool(existing.get("locked", False)),
                    "state": "locked" if existing.get("locked") else "planned",
                }
            )
        return generated

    def _build_scenes_from_plan(
        self,
        manifest: Dict[str, Any],
        scene_plan: List[Any],
        *,
        default_scene_type: Any = None,
    ) -> List[Dict[str, Any]]:
        lines: List[str] = []
        types: List[str] = []
        for item in scene_plan:
            if isinstance(item, str):
                goal = item.strip()
                if goal:
                    lines.append(goal)
                    types.append(self._normalize_scene_type(default_scene_type))
                continue
            if not isinstance(item, dict):
                continue
            goal = str(
                item.get("scene_goal") or item.get("goal") or item.get("description") or ""
            ).strip()
            if not goal:
                continue
            lines.append(goal)
            types.append(
                self._normalize_scene_type(
                    item.get("scene_type")
                    or default_scene_type
                    or self._infer_scene_type_from_goal(goal)
                )
            )
        if not lines:
            return manifest.get("scenes", [])
        scenes = self._build_scenes_from_text(
            manifest,
            "\n".join(lines),
            default_scene_type=default_scene_type,
        )
        for idx, scene in enumerate(scenes):
            if idx < len(types):
                scene["scene_type"] = types[idx]
        return scenes

    def _normalize_scene_entries(
        self,
        scenes: List[Any],
        existing_scenes: List[Any],
        *,
        default_scene_type: Any = None,
    ) -> List[Dict[str, Any]]:
        existing_by_id = {
            item.get("scene_id"): item
            for item in existing_scenes
            if isinstance(item, dict) and item.get("scene_id")
        }
        normalized: List[Dict[str, Any]] = []
        for index, scene in enumerate(scenes, start=1):
            if not isinstance(scene, dict):
                continue
            provided_id = str(scene.get("scene_id") or "").strip()
            scene_id = provided_id or f"scene_{index:03d}"
            existing = existing_by_id.get(scene_id, {})
            merged = dict(existing)
            merged.update(scene)
            merged["scene_id"] = scene_id
            merged["scene_prompt"] = str(
                merged.get("scene_goal") or merged.get("scene_prompt") or ""
            ).strip()
            merged["scene_goal"] = str(
                merged.get("scene_goal") or merged.get("scene_prompt") or ""
            ).strip()
            if not merged["scene_goal"]:
                continue
            if not merged.get("conversation") and merged.get("dialogue"):
                merged["conversation"] = [
                    f"{str(line.get('speaker_id') or 'narrator')}: {str(line.get('line') or '').strip()}"
                    for line in merged.get("dialogue", [])
                    if isinstance(line, dict) and str(line.get("line", "")).strip()
                ]
            merged["scene_type"] = self._normalize_scene_type(
                merged.get("scene_type")
                or default_scene_type
                or self._infer_scene_type_from_goal(merged["scene_goal"])
            )
            merged.setdefault("location_time", existing.get("location_time", ""))
            merged.setdefault("characters_in_scene", existing.get("characters_in_scene", []))
            merged.setdefault("events", existing.get("events", []))
            merged.setdefault("dialogue", existing.get("dialogue", []))
            merged.setdefault("narration", existing.get("narration", ""))
            merged.setdefault("shot_list", existing.get("shot_list", []))
            merged["locked"] = bool(merged.get("locked", existing.get("locked", False)))
            if merged["locked"]:
                merged["state"] = "locked"
            elif merged.get("state") not in self.SCENE_STATES:
                merged["state"] = existing.get("state", "planned")
            normalized.append(merged)
        return normalized

    def _normalize_character_entries(self, characters: List[Any]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for index, item in enumerate(characters, start=1):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            role = str(item.get("role", "")).strip()
            appearance = str(item.get("appearance", "")).strip()
            personality = str(item.get("personality", "")).strip()
            if not any([name, role, appearance, personality]):
                continue
            character_id = self._slug_character_id(
                str(item.get("character_id", "")).strip(),
                name,
                index,
            )
            voice = item.get("voice", {})
            if not isinstance(voice, dict):
                voice = {}
            voice_provider = str(voice.get("voice_provider", "native")).strip() or "native"
            voice_id = str(voice.get("voice_id", "")).strip() or f"native_{character_id}"
            speaking_style = (
                str(voice.get("speaking_style", "")).strip() or "clear, natural, medium pace"
            )
            normalized.append(
                {
                    "character_id": character_id,
                    "name": name or f"Character {index}",
                    "role": role or "protagonist",
                    "appearance": appearance or "Age 20-40, neutral styling",
                    "personality": personality or "confident",
                    "constraints": str(
                        item.get("constraints", "Avoid resemblance to real people.")
                    ).strip(),
                    "voice": {
                        "voice_provider": voice_provider,
                        "voice_id": voice_id,
                        "speaking_style": speaking_style,
                    },
                    "reference_images": item.get("reference_images", []),
                }
            )
        return normalized

    @staticmethod
    def _split_text_items(value: str) -> List[str]:
        items: List[str] = []
        for raw in str(value).replace(",", "\n").splitlines():
            entry = raw.strip()
            if entry:
                items.append(entry)
        return items

    def _normalize_scene_type(self, value: Any) -> str:
        text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        if not text:
            return "auto"
        if text in self.SCENE_TYPES:
            return text
        aliases = {
            "hook_scene": "hook",
            "problem_scene": "problem",
            "reframe_scene": "reframe",
            "solution_scene": "solution",
            "payoff_scene": "payoff",
            "call_to_action": "cta",
            "closing_cta": "cta",
            "broll": "b_roll",
            "b_roll_scene": "b_roll",
        }
        normalized = aliases.get(text, "")
        return normalized if normalized in self.SCENE_TYPES else "custom"

    def _infer_scene_type_from_goal(self, goal: str) -> str:
        text = str(goal or "").lower()
        rules = [
            ("hook", "hook"),
            ("problem", "problem"),
            ("pain", "problem"),
            ("reframe", "reframe"),
            ("shift", "reframe"),
            ("solution", "solution"),
            ("how it works", "solution"),
            ("payoff", "payoff"),
            ("benefit", "payoff"),
            ("cta", "cta"),
            ("call to action", "cta"),
            ("download", "cta"),
            ("testimonial", "testimonial"),
            ("tutorial", "tutorial"),
            ("b-roll", "b_roll"),
            ("b roll", "b_roll"),
            ("transition", "transition"),
        ]
        for keyword, scene_type in rules:
            if keyword in text:
                return scene_type
        return "auto"

    @staticmethod
    def _infer_scene_count_hint(text: str) -> int:
        numbers = []
        for match in re.finditer(r"\bscene\s*(\d{1,2})\b", str(text), flags=re.I):
            try:
                numbers.append(int(match.group(1)))
            except Exception:
                continue
        if numbers:
            return max(numbers)
        return 0

    def _extract_scene_plan_from_description(self, text: str) -> List[Dict[str, str]]:
        lines: List[Dict[str, str]] = []
        pattern = re.compile(
            r"^\s*(?:[^\w\s]*)?\s*scene\s*(\d{1,2})\s*[:\-]\s*(.+)$",
            flags=re.I | re.M,
        )
        for match in pattern.finditer(str(text)):
            goal = re.sub(r"\s+", " ", match.group(2)).strip()
            if not goal:
                continue
            scene_type = self._infer_scene_type_from_goal(goal)
            lines.append({"scene_goal": goal, "scene_type": scene_type})
        return lines

    @staticmethod
    def _extract_primary_character_name(text: str) -> str:
        if not text:
            return ""
        pattern = re.compile(
            r"\bname\s*:\s*([A-Z][a-zA-Z][a-zA-Z' -]{1,40})",
            flags=re.I,
        )
        match = pattern.search(str(text))
        if match:
            return match.group(1).strip()
        profile_pattern = re.compile(
            r"\bcharacter\s+profile\s*:\s*([A-Z][a-zA-Z][a-zA-Z' -]{1,40})",
            flags=re.I,
        )
        profile_match = profile_pattern.search(str(text))
        if profile_match:
            return profile_match.group(1).strip()
        return ""

    @staticmethod
    def _slug_character_id(existing_id: str, name: str, index: int) -> str:
        base = existing_id.strip() if existing_id else name.strip().lower()
        if not base:
            return f"char_{index}"
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", base).strip("_").lower()
        if not slug:
            slug = f"char_{index}"
        if not slug.startswith("char_"):
            slug = f"char_{slug}"
        return slug

    def _ensure_project_layout(self, project_id: str) -> None:
        project_dir = self._project_dir(project_id)
        (project_dir / "assets" / "video").mkdir(parents=True, exist_ok=True)
        (project_dir / "assets" / "audio").mkdir(parents=True, exist_ok=True)
        (project_dir / "assets" / "subtitles").mkdir(parents=True, exist_ok=True)
        (project_dir / "assets" / "references" / "characters").mkdir(
            parents=True,
            exist_ok=True,
        )
        (project_dir / "reports").mkdir(parents=True, exist_ok=True)
        (project_dir / "jobs").mkdir(parents=True, exist_ok=True)
        (project_dir / "snapshots").mkdir(parents=True, exist_ok=True)
        (project_dir / "exports").mkdir(parents=True, exist_ok=True)

    def _project_dir(self, project_id: str) -> Path:
        return self.projects_root / project_id

    def _manifest_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "manifest.json"

    def _autofill_log_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "autofill_log.json"

    def _job_path(self, project_id: str, job_id: str) -> Path:
        self._ensure_project_layout(project_id)
        return self._project_dir(project_id) / "jobs" / f"{job_id}.json"

    def _snapshot_dir(self, project_id: str) -> Path:
        self._ensure_project_layout(project_id)
        return self._project_dir(project_id) / "snapshots"

    def _snapshot_path(self, project_id: str, snapshot_id: str) -> Path:
        return self._snapshot_dir(project_id) / f"{snapshot_id}.json"

    def _export_dir(self, project_id: str) -> Path:
        self._ensure_project_layout(project_id)
        return self._project_dir(project_id) / "exports"

    @staticmethod
    def _find_scene(manifest: Dict[str, Any], scene_id: str) -> Optional[Dict[str, Any]]:
        for scene in manifest.get("scenes", []):
            if isinstance(scene, dict) and scene.get("scene_id") == scene_id:
                return scene
        return None

    @staticmethod
    def _scene_duration_seconds(scene: Dict[str, Any]) -> float:
        shots = scene.get("shot_list") or []
        if shots:
            return max(0.0, float(sum(float(shot.get("duration_seconds", 0)) for shot in shots)))
        dialogue = scene.get("dialogue") or []
        if dialogue:
            try:
                start = float(dialogue[0].get("start_second", 0))
                end = float(dialogue[-1].get("end_second", start))
                return max(0.0, end - start)
            except Exception:
                return 0.0
        return 0.0

    @staticmethod
    def _rebuild_timeline_entries(manifest: Dict[str, Any]) -> None:
        timeline = manifest.setdefault("timeline", {})
        entries: List[Dict[str, Any]] = []
        for scene in manifest.get("scenes", []):
            scene_id = scene.get("scene_id")
            for line in scene.get("dialogue", []):
                entries.append(
                    {
                        "scene_id": scene_id,
                        "speaker_id": line.get("speaker_id"),
                        "line": line.get("line", ""),
                        "start_second": float(line.get("start_second", 0.0)),
                        "end_second": float(line.get("end_second", 0.0)),
                    }
                )
        entries.sort(key=lambda item: (item.get("start_second", 0.0), item.get("scene_id", "")))
        timeline["entries"] = entries

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        with ManifestService._io_lock:
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with ManifestService._io_lock:
            temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            try:
                with open(temp_path, "w", encoding="utf-8") as file:
                    json.dump(data, file, indent=2, ensure_ascii=False, sort_keys=True)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temp_path, path)
            finally:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)
