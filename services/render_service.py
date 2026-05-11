from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from google.genai import types

from services.autofill_service import ManifestAutoFillService
from services.manifest_service import ManifestService
from services.report_service import ReportService
from services.safety_service import SafetyCheckService
from services.validation_service import ValidationService
from services.veo_client_service import VeoClientService
from services.voice_service import VoiceService


ProgressCallback = Optional[Callable[[str, float, Optional[str]], None]]


class RenderCancelled(Exception):
    pass


class RenderService:
    def __init__(
        self,
        manifest_service: ManifestService,
        validation_service: ValidationService,
        autofill_service: ManifestAutoFillService,
        safety_service: SafetyCheckService,
        voice_service: VoiceService,
        report_service: ReportService,
        veo_client_service: VeoClientService,
        enable_real_generation: bool = True,
        pause_poll_seconds: float = 1.0,
    ):
        self.manifest_service = manifest_service
        self.validation_service = validation_service
        self.autofill_service = autofill_service
        self.safety_service = safety_service
        self.voice_service = voice_service
        self.report_service = report_service
        self.veo_client_service = veo_client_service
        self.enable_real_generation = enable_real_generation
        self.pause_poll_seconds = pause_poll_seconds

    def render_project(
        self,
        project_id: str,
        job_id: str,
        progress_cb: ProgressCallback = None,
        target_scene_id: Optional[str] = None,
        scene_only: bool = False,
    ) -> Dict[str, Any]:
        scene_only_mode = bool(scene_only and target_scene_id)
        manifest = self.manifest_service.get_manifest(project_id)
        manifest = self.validation_service.normalize_manifest(manifest)
        existing_approvals = dict(manifest.get("approvals", {}))
        if not scene_only_mode:
            script_was_approved = bool(existing_approvals.get("script_approved"))
            manifest = self.autofill_service.autofill_all(
                manifest, reason="render_finalize_autofill"
            )
            manifest = self.validation_service.normalize_manifest(manifest)
            if script_was_approved:
                approvals = manifest.setdefault("approvals", {})
                approvals["script_approved"] = True
                approvals["script_needs_user_approval"] = False
                if existing_approvals.get("script_last_reviewed_at"):
                    approvals["script_last_reviewed_at"] = existing_approvals.get(
                        "script_last_reviewed_at"
                    )
                if existing_approvals.get("script_reviewer_note"):
                    approvals["script_reviewer_note"] = existing_approvals.get(
                        "script_reviewer_note"
                    )
            self.validation_service.ensure_render_eligible(manifest)

        if target_scene_id and not any(
            isinstance(scene, dict) and scene.get("scene_id") == target_scene_id
            for scene in manifest.get("scenes", [])
        ):
            raise FileNotFoundError(f"Scene '{target_scene_id}' not found in project '{project_id}'.")
        generation_mode = str(manifest.get("generation_mode", "auto")).lower()
        manifest["render_iteration"] = int(manifest.get("render_iteration", 0) or 0) + 1
        for scene in manifest.get("scenes", []):
            if scene.get("locked"):
                scene["state"] = "locked"
            elif scene.get("state") == "stitched":
                scene["state"] = "approved"
            elif scene.get("state") not in {"planned", "generated", "approved", "locked"}:
                scene["state"] = "planned"

        warnings = self.safety_service.run_checks(manifest)
        manifest["safety_warnings"] = warnings
        self.manifest_service.save_manifest(project_id, manifest)

        paths = self.manifest_service.project_paths(project_id)
        video_dir = paths["assets_video_dir"]
        audio_dir = paths["assets_audio_dir"]
        subtitle_dir = paths["assets_subtitles_dir"]
        frames_dir = os.path.join(video_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        pipeline_warnings: List[Dict[str, Any]] = list(warnings)
        clip_records: List[Dict[str, str]] = []
        continuity_frame_path: Optional[str] = None
        if scene_only_mode and target_scene_id:
            continuity_frame_path = self._seed_continuity_from_previous_scene(
                manifest=manifest,
                project_id=project_id,
                target_scene_id=target_scene_id,
                project_dir=paths["project_dir"],
                video_dir=video_dir,
                frames_dir=frames_dir,
                warnings=pipeline_warnings,
            )

        if progress_cb:
            progress_cb("preparing", 0.05, None)

        scene_index = 0
        while True:
            self._honor_job_controls(project_id, job_id, progress_cb, scene_hint=None)
            manifest = self.manifest_service.get_manifest(project_id)
            manifest = self.validation_service.normalize_manifest(manifest)
            scenes = manifest.get("scenes", [])
            if scene_index >= len(scenes):
                break

            scene = scenes[scene_index]
            scene_id = scene.get("scene_id", f"scene_{scene_index + 1:03d}")
            if scene_only_mode and target_scene_id and scene_id != target_scene_id:
                scene_index += 1
                continue

            if scene_only_mode:
                scene_phase_start = 0.10
                scene_phase_span = 0.70
            else:
                scene_total = max(1, len(scenes))
                scene_phase_start = 0.10 + (0.40 * (scene_index / scene_total))
                scene_phase_span = 0.40 / scene_total
            self._update_scene_generation_progress(
                project_id=project_id,
                job_id=job_id,
                scene_id=scene_id,
                scene_progress=0.0,
                scene_phase_start=scene_phase_start,
                scene_phase_span=scene_phase_span,
                progress_cb=progress_cb,
            )
            self._update_scene_generation_progress(
                project_id=project_id,
                job_id=job_id,
                scene_id=scene_id,
                scene_progress=0.08,
                scene_phase_start=scene_phase_start,
                scene_phase_span=scene_phase_span,
                progress_cb=progress_cb,
            )

            output_path = os.path.join(
                video_dir, f"{project_id}_{scene_id}_{scene_index + 1:03d}.mp4"
            )
            continuity_mode, continuity_reason = self._should_use_continuity_anchor(
                scene_index=scene_index,
                scene=scene,
                scenes=scenes,
            )
            active_continuity_frame_path = (
                continuity_frame_path if continuity_mode else None
            )
            self._ensure_continuity_input_available(
                scene_index=scene_index,
                scene_id=scene_id,
                scene=scene,
                continuity_frame_path=active_continuity_frame_path,
                require_continuity=continuity_mode,
            )
            prompt = self._build_scene_prompt(
                manifest,
                scene,
                active_continuity_frame_path,
            )
            reference_inputs = self._collect_reference_image_paths(
                manifest,
                scene,
                active_continuity_frame_path,
                project_dir=paths["project_dir"],
            )
            selected_reference_inputs = self._select_reference_inputs_for_generation(
                reference_inputs=reference_inputs,
                scene_index=scene_index,
                continuity_mode=continuity_mode,
            )
            refs = self._build_reference_images(
                reference_inputs=selected_reference_inputs
            )
            # Keep continuity image loading validation, but send it as a regular
            # reference image so every scene request follows the same flow.
            last_frame_image: Optional[types.Image] = self._build_last_frame_image(
                reference_inputs=selected_reference_inputs
            )
            generation_aspect_ratio = self._resolve_generation_aspect_ratio(
                manifest,
                has_reference_images=bool(refs),
                continuity_frame_path=active_continuity_frame_path,
            )
            self._ensure_last_frame_anchor_ready(
                scene_index=scene_index,
                scene_id=scene_id,
                scene=scene,
                continuity_frame_path=active_continuity_frame_path,
                last_frame_image=last_frame_image,
                require_continuity=continuity_mode,
            )

            scene_debug = scene.setdefault("render_debug", {})
            scene_debug["veo_prompt"] = prompt
            scene_debug["continuity_mode"] = continuity_mode
            scene_debug["continuity_mode_reason"] = continuity_reason
            scene_debug["reference_images"] = [
                self._to_project_relative_path(item.get("path", ""), paths["project_dir"])
                for item in reference_inputs
            ]
            scene_debug["reference_images_sent"] = [
                self._to_project_relative_path(item.get("path", ""), paths["project_dir"])
                for item in selected_reference_inputs
            ]
            continuity_candidates = [
                item
                for item in selected_reference_inputs
                if str(item.get("role", "")).strip().lower() == "continuity"
            ]
            scene_debug["last_frame_sent"] = ""
            scene_debug["continuity_reference_sent"] = (
                self._to_project_relative_path(
                    continuity_candidates[0].get("path", ""),
                    paths["project_dir"],
                )
                if continuity_candidates
                else ""
            )
            scene_debug["reference_inputs"] = [
                {
                    "path": self._to_project_relative_path(
                        item.get("path", ""),
                        paths["project_dir"],
                    ),
                    "role": item.get("role", ""),
                }
                for item in reference_inputs
            ]
            scene_debug["scene_builder_snapshot"] = {
                "scene_id": scene.get("scene_id"),
                "scene_type": scene.get("scene_type"),
                "scene_goal": scene.get("scene_goal"),
                "scene_description": scene.get("scene_description"),
                "spoken_direction": scene.get("spoken_direction"),
                "image_usage_instructions": scene.get("image_usage_instructions"),
                "location_time": scene.get("location_time"),
                "characters_in_scene": list(scene.get("characters_in_scene", []) or []),
                "events": list(scene.get("events", []) or []),
                "continuity": dict(scene.get("continuity", {}) or {}),
            }
            scene_debug["generation_status"] = "started"
            self.manifest_service.save_manifest(project_id, manifest)

            try:
                if self.enable_real_generation:
                    video_bytes = self.veo_client_service.generate_video_bytes(
                        prompt=prompt,
                        reference_images=refs,
                        aspect_ratio=generation_aspect_ratio,
                        last_frame=None,
                        allow_reference_fallback=False,
                        progress_callback=lambda progress_value: self._update_scene_generation_progress(
                            project_id=project_id,
                            job_id=job_id,
                            scene_id=scene_id,
                            scene_progress=0.12 + (0.76 * float(progress_value)),
                            scene_phase_start=scene_phase_start,
                            scene_phase_span=scene_phase_span,
                            progress_cb=progress_cb,
                        ),
                    )
                    with open(output_path, "wb") as file:
                        file.write(video_bytes)
                else:
                    self._create_placeholder_clip(output_path, scene_id, scene)
            except Exception as error:
                error_message = str(error)
                if self._is_quota_exhausted_error(error_message):
                    scene_debug["generation_status"] = "failed_quota"
                    scene_debug["generation_error"] = error_message
                    self.manifest_service.save_manifest(project_id, manifest)
                    raise RuntimeError(
                        "Veo API quota exhausted (429 RESOURCE_EXHAUSTED). "
                        "Render stopped after first failure. "
                        "Restore quota/billing and rerun render."
                    ) from error

                if continuity_mode:
                    scene_debug["generation_status"] = "failed_continuity"
                    scene_debug["generation_error"] = error_message
                    self.manifest_service.save_manifest(project_id, manifest)
                    raise RuntimeError(
                        f"Continuity generation failed for {scene_id}: {error_message}"
                    ) from error

                scene_debug["generation_status"] = "failed_reference_lock"
                scene_debug["generation_error"] = error_message
                self.manifest_service.save_manifest(project_id, manifest)
                self._create_placeholder_clip(output_path, scene_id, scene)
                pipeline_warnings.append(
                    {
                        "category": "generation_fallback",
                        "severity": "warning",
                        "message": f"Used placeholder clip for {scene_id}: {error_message}",
                        "field_path": f"scenes.{scene_id}",
                    }
                )
            output_path = self._ensure_web_playable_clip(output_path, pipeline_warnings)
            scene_debug["output_clip_path"] = self._to_project_relative_path(
                output_path,
                paths["project_dir"],
            )
            scene_debug["generation_status"] = "generated"
            self.manifest_service.save_manifest(project_id, manifest)
            self._update_scene_generation_progress(
                project_id=project_id,
                job_id=job_id,
                scene_id=scene_id,
                scene_progress=0.92,
                scene_phase_start=scene_phase_start,
                scene_phase_span=scene_phase_span,
                progress_cb=progress_cb,
            )

            end_frame_path = os.path.join(
                frames_dir, f"{project_id}_{scene_id}_{scene_index + 1:03d}_end.jpg"
            )
            extracted_end_frame_path: Optional[str] = None
            has_follow_up_scene = (not scene_only_mode) and ((scene_index + 1) < len(scenes))
            try:
                self._extract_last_frame(output_path, end_frame_path)
                extracted_end_frame_path = end_frame_path
            except Exception as error:
                continuity_frame_path = None
                scene_debug["continuity_frame_status"] = "failed_extract"
                scene_debug["continuity_frame_error"] = str(error)
                self.manifest_service.save_manifest(project_id, manifest)
                if has_follow_up_scene:
                    raise RuntimeError(
                        f"Continuity chain stopped at {scene_id}: failed to extract last frame ({error}). "
                        "Generation stopped to avoid reusing the initial reference image."
                    ) from error
                pipeline_warnings.append(
                    {
                        "category": "continuity_frame_failure",
                        "severity": "warning",
                        "message": f"Could not extract continuity frame for {scene_id}: {error}",
                        "field_path": f"scenes.{scene_id}",
                    }
                )

            if extracted_end_frame_path:
                continuity_frame_path = extracted_end_frame_path
                scene_debug["continuity_frame_status"] = "ready"
                scene_debug["continuity_frame_path"] = self._to_project_relative_path(
                    extracted_end_frame_path,
                    paths["project_dir"],
                )

            clip_records.append(
                {
                    "scene_id": scene_id,
                    "clip_path": output_path.replace("\\", "/"),
                    "end_frame_path": extracted_end_frame_path.replace("\\", "/")
                    if extracted_end_frame_path
                    else "",
                }
            )
            self._update_scene_generation_progress(
                project_id=project_id,
                job_id=job_id,
                scene_id=scene_id,
                scene_progress=1.0,
                scene_phase_start=scene_phase_start,
                scene_phase_span=scene_phase_span,
                progress_cb=progress_cb,
            )

            try:
                self.manifest_service.set_scene_state(
                    project_id,
                    scene_id,
                    "generated",
                    note="Scene clip generated.",
                    force=True,
                )
            except Exception:
                pipeline_warnings.append(
                    {
                        "category": "scene_state_update_failure",
                        "severity": "warning",
                        "message": f"Could not mark {scene_id} as generated.",
                        "field_path": f"scenes.{scene_id}.state",
                    }
                )

            if generation_mode == "auto":
                try:
                    self.manifest_service.set_scene_state(
                        project_id,
                        scene_id,
                        "approved",
                        note="Auto mode approved scene after generation.",
                        force=True,
                    )
                except Exception:
                    pipeline_warnings.append(
                        {
                            "category": "scene_auto_approve_failure",
                            "severity": "warning",
                            "message": f"Could not auto-approve {scene_id}.",
                            "field_path": f"scenes.{scene_id}.state",
                        }
                    )
            else:
                if not scene_only_mode:
                    self._wait_for_scene_approval(
                        project_id,
                        job_id,
                        progress_cb,
                        scene_id=scene_id,
                    )

            scene_index += 1
            if scene_only_mode:
                break

        if scene_only_mode:
            manifest = self.manifest_service.get_manifest(project_id)
            manifest["safety_warnings"] = pipeline_warnings
            self.manifest_service.save_manifest(project_id, manifest)
            artifacts = [
                {
                    "type": "clip",
                    "path": record.get("clip_path", "").replace("\\", "/"),
                }
                for record in clip_records
                if record.get("clip_path")
            ]
            for record in clip_records:
                if record.get("end_frame_path"):
                    artifacts.append(
                        {
                            "type": "continuity_frame",
                            "path": record.get("end_frame_path", "").replace("\\", "/"),
                        }
                    )
            if progress_cb:
                progress_cb("completed", 1.0, target_scene_id)
            return {
                "project_id": project_id,
                "job_id": job_id,
                "status": "completed",
                "warnings": pipeline_warnings,
                "artifacts": artifacts,
                "scene_only": True,
                "scene_id": target_scene_id,
            }

        if progress_cb:
            progress_cb("audio_generation", 0.58, None)
        self.manifest_service.update_job(
            project_id,
            job_id,
            current_scene_id=None,
            current_scene_progress=0.0,
        )
        voice_tracks = self.voice_service.synthesize_dialogue_tracks(manifest, audio_dir)
        manifest.setdefault("audio", {})
        manifest["audio"]["voice_tracks"] = voice_tracks
        manifest["audio"]["voice_qa"] = [
            {
                "scene_id": item.get("scene_id"),
                "speaker_id": item.get("speaker_id"),
                "line": item.get("line"),
                "qa": item.get("qa", {}),
            }
            for item in voice_tracks
        ]
        self.manifest_service.save_manifest(project_id, manifest)

        subtitle_path = ""
        if manifest.get("output", {}).get("subtitles_enabled", True):
            subtitle_entries = self._build_subtitle_entries(manifest)
            subtitle_path = os.path.join(subtitle_dir, f"{project_id}_captions.srt")
            self.report_service.write_subtitle_file(subtitle_entries, subtitle_path)

        if progress_cb:
            progress_cb("stitching", 0.74, None)

        final_video_path = os.path.join(video_dir, f"{project_id}_final.mp4")
        final_video_path = self._stitch_final_video(
            clip_records=clip_records,
            voice_tracks=voice_tracks,
            subtitle_path=subtitle_path,
            output_path=final_video_path,
            output_spec=manifest.get("output", {}),
            timeline=manifest.get("timeline", {}),
            warnings=pipeline_warnings,
        )

        artifacts = self._write_reports_and_artifacts(
            manifest=manifest,
            paths=paths,
            final_video_path=final_video_path,
            clip_records=clip_records,
            voice_tracks=voice_tracks,
            subtitle_path=subtitle_path,
            warnings=pipeline_warnings,
        )

        for scene in manifest.get("scenes", []):
            scene_id = scene.get("scene_id")
            if not scene_id:
                continue
            try:
                self.manifest_service.set_scene_state(
                    project_id,
                    scene_id,
                    "stitched",
                    note="Final video stitched.",
                    force=True,
                )
            except Exception:
                pipeline_warnings.append(
                    {
                        "category": "scene_stitched_state_failure",
                        "severity": "warning",
                        "message": f"Could not mark {scene_id} as stitched.",
                        "field_path": f"scenes.{scene_id}.state",
                    }
                )

        manifest = self.manifest_service.get_manifest(project_id)
        manifest["safety_warnings"] = pipeline_warnings
        self.manifest_service.save_manifest(project_id, manifest)

        if progress_cb:
            progress_cb("completed", 1.0, None)

        return {
            "project_id": project_id,
            "job_id": job_id,
            "status": "completed",
            "warnings": pipeline_warnings,
            "artifacts": artifacts,
            "final_video_path": final_video_path.replace("\\", "/"),
        }

    def _update_scene_generation_progress(
        self,
        project_id: str,
        job_id: str,
        scene_id: str,
        scene_progress: float,
        scene_phase_start: float,
        scene_phase_span: float,
        progress_cb: ProgressCallback,
    ) -> None:
        normalized = max(0.0, min(1.0, float(scene_progress)))
        overall_progress = scene_phase_start + (scene_phase_span * normalized)
        self.manifest_service.update_job(
            project_id,
            job_id,
            status="running",
            progress=overall_progress,
            current_stage="scene_generation",
            current_scene_id=scene_id,
            current_scene_progress=normalized,
        )
        if progress_cb:
            progress_cb("scene_generation", overall_progress, scene_id)

    def _wait_for_script_approval(
        self,
        project_id: str,
        job_id: str,
        progress_cb: ProgressCallback,
        scene_id: str,
    ) -> None:
        while True:
            self._honor_job_controls(project_id, job_id, progress_cb, scene_hint=scene_id)
            manifest = self.manifest_service.get_manifest(project_id)
            approved = bool(manifest.get("approvals", {}).get("script_approved"))
            if approved:
                return
            self.manifest_service.update_job(
                project_id,
                job_id,
                status="paused",
                current_stage="awaiting_script_approval",
                current_scene_id=scene_id,
                control_note="Waiting for script approval.",
            )
            time.sleep(self.pause_poll_seconds)

    def _wait_for_scene_approval(
        self,
        project_id: str,
        job_id: str,
        progress_cb: ProgressCallback,
        scene_id: str,
    ) -> None:
        while True:
            self._honor_job_controls(project_id, job_id, progress_cb, scene_hint=scene_id)
            manifest = self.manifest_service.get_manifest(project_id)
            matched_scene = None
            for scene in manifest.get("scenes", []):
                if scene.get("scene_id") == scene_id:
                    matched_scene = scene
                    break
            if matched_scene is None:
                return
            if str(matched_scene.get("state", "planned")) in {"approved", "locked", "stitched"}:
                return
            self.manifest_service.update_job(
                project_id,
                job_id,
                status="paused",
                current_stage="awaiting_scene_approval",
                current_scene_id=scene_id,
                control_note="Directed mode waiting for scene approval.",
            )
            time.sleep(self.pause_poll_seconds)

    def _honor_job_controls(
        self,
        project_id: str,
        job_id: str,
        progress_cb: ProgressCallback,
        scene_hint: Optional[str],
    ) -> None:
        while True:
            job = self.manifest_service.get_job(job_id)
            if job.get("cancel_requested"):
                self.manifest_service.update_job(
                    project_id,
                    job_id,
                    status="cancelled",
                    current_stage="cancelled",
                    current_scene_id=scene_hint,
                    control_note="Cancelled by user.",
                )
                raise RenderCancelled("Render cancelled by user.")

            if job.get("pause_requested"):
                self.manifest_service.update_job(
                    project_id,
                    job_id,
                    status="paused",
                    current_stage="paused",
                    current_scene_id=scene_hint,
                    control_note="Paused by user.",
                )
                if progress_cb:
                    progress_cb("paused", float(job.get("progress", 0.0)), scene_hint)
                time.sleep(self.pause_poll_seconds)
                continue

            if job.get("status") == "paused":
                self.manifest_service.update_job(
                    project_id,
                    job_id,
                    status="running",
                    current_stage="running",
                    current_scene_id=scene_hint,
                    control_note="Resumed.",
                )
            return

    def _build_scene_prompt(
        self,
        manifest: Dict[str, Any],
        scene: Dict[str, Any],
        continuity_frame_path: Optional[str],
    ) -> str:
        bible = manifest.get("character_bible", {})
        characters_by_id = {
            item.get("character_id"): item
            for item in manifest.get("characters", [])
            if isinstance(item, dict) and item.get("character_id")
        }
        lines: List[str] = []
        continuity_mode = bool(continuity_frame_path)
        scene_goal = str(scene.get("scene_goal", "") or "").strip()
        if scene_goal.lower().startswith("continue from scene_"):
            parts = scene_goal.split(".", 1)
            if len(parts) == 2:
                scene_goal = parts[1].strip()
        scene_description = str(scene.get("scene_description", "") or "").strip()
        spoken_direction = str(scene.get("spoken_direction", "") or "").strip()
        image_usage = str(scene.get("image_usage_instructions", "") or "").strip()
        events = [
            str(item).strip()
            for item in scene.get("events", [])
            if str(item).strip()
        ]

        if continuity_mode:
            lines.append("Generate one continuation shot from the supplied reference image.")
            lines.append(
                "Frame 1 must match the reference image for identity, wardrobe, camera angle, and room layout."
            )
        else:
            lines.append("Generate one opening shot using supplied reference image(s).")

        lines.append(f"Scene type: {scene.get('scene_type', 'auto')}")
        if scene_goal:
            lines.append(f"Action: {scene_goal}")
        elif scene_description:
            lines.append(f"Action: {scene_description}")

        if scene_description and scene_description != scene_goal:
            lines.append(f"Visual: {scene_description}")
        if spoken_direction:
            lines.append(f"Performance: {spoken_direction}")
        elif image_usage:
            lines.append(f"Performance: {image_usage}")
        if events:
            lines.append("Key beats: " + "; ".join(events[:2]))

        scene_cast = [
            cid
            for cid in scene.get("characters_in_scene", [])
            if isinstance(cid, str) and cid.strip()
        ]
        if scene_cast:
            cast_labels = []
            for cid in scene_cast:
                char = characters_by_id.get(cid, {})
                cast_labels.append(f"{cid} ({char.get('name', cid)})")
            lines.append("On-screen cast: " + ", ".join(cast_labels))

        for character_id in scene.get("characters_in_scene", []):
            char = bible.get(character_id, {})
            if not isinstance(char, dict):
                continue
            lines.append(
                f"{character_id} lock: keep face identity, skin tone, hairstyle, and wardrobe consistent."
            )

        lines.append("Keep the shot realistic and natural.")
        return "\n".join(lines)

    def _collect_reference_image_paths(
        self,
        manifest: Dict[str, Any],
        scene: Dict[str, Any],
        continuity_frame_path: Optional[str],
        project_dir: str = "",
    ) -> List[Dict[str, str]]:
        seen = set()
        resolved_inputs: List[Dict[str, str]] = []

        def add_candidate(raw_path: str, role: str) -> None:
            if not isinstance(raw_path, str):
                return
            normalized = raw_path.strip()
            if not normalized or normalized in seen:
                return
            normalized_path = normalized.replace("/", os.sep)
            resolved_path = normalized_path
            if not os.path.isabs(resolved_path):
                # Some stored paths are already repo-root-relative (for example
                # data/projects/.../assets/video/frames/...); do not prepend the
                # project directory a second time in that case.
                if not os.path.isfile(resolved_path) and project_dir:
                    resolved_path = os.path.join(project_dir, normalized_path)
            if not os.path.isfile(resolved_path):
                return
            seen.add(normalized)
            resolved_inputs.append({"path": resolved_path, "role": role})

        # Prioritize character identity refs before continuity/style refs.
        bible = manifest.get("character_bible", {})
        for character_id in scene.get("characters_in_scene", []):
            entry = bible.get(character_id, {})
            if isinstance(entry, dict):
                for ref in entry.get("reference_images", []):
                    add_candidate(ref, "character")

        for ref in scene.get("reference_images", []):
            add_candidate(ref, "scene")

        if continuity_frame_path:
            add_candidate(continuity_frame_path, "continuity")

        world = manifest.get("world", {})
        for ref in world.get("visual_references", []):
            add_candidate(ref, "world_style")

        # Keep request size stable and deterministic.
        return resolved_inputs[:6]

    def _build_reference_images(
        self,
        reference_inputs: List[Dict[str, str]],
    ) -> List[types.VideoGenerationReferenceImage]:
        refs: List[types.VideoGenerationReferenceImage] = []
        for item in reference_inputs:
            path = str(item.get("path", "")).strip()
            try:
                image_payload = self._build_image_from_path(path)
                refs.append(
                    types.VideoGenerationReferenceImage(image=image_payload)
                )
            except Exception:
                continue
            if refs:
                break
        return refs

    @staticmethod
    def _is_quota_exhausted_error(message: str) -> bool:
        lowered = (message or "").lower()
        return "resource_exhausted" in lowered or "quota" in lowered or "429" in lowered

    def _select_reference_inputs_for_generation(
        self,
        reference_inputs: List[Dict[str, str]],
        scene_index: int,
        continuity_mode: bool = True,
    ) -> List[Dict[str, str]]:
        # Workflow rule:
        # - Scene 1: send one primary uploaded reference image.
        # - Scene 2+ with continuity: send previous scene end-frame only.
        # - Scene 2+ without continuity: treat as opening-style request.
        if scene_index > 0 and continuity_mode:
            for item in reference_inputs:
                if str(item.get("role", "")).strip().lower() == "continuity":
                    return [item]
            return []

        # Mirror app.py: keep references simple and stable; send one primary image.
        priority = ("character", "scene", "world_style")
        for role in priority:
            for item in reference_inputs:
                if str(item.get("role", "")).strip().lower() == role:
                    return [item]

        for item in reference_inputs:
            if str(item.get("role", "")).strip().lower() != "continuity":
                return [item]
        return []

    @staticmethod
    def _scene_cast_ids(scene: Dict[str, Any]) -> List[str]:
        return [
            cid
            for cid in scene.get("characters_in_scene", [])
            if isinstance(cid, str) and cid.strip()
        ]

    def _should_use_continuity_anchor(
        self,
        *,
        scene_index: int,
        scene: Dict[str, Any],
        scenes: List[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        if not self._scene_requires_continuity(scene_index=scene_index, scene=scene):
            return False, "continuity_not_required"
        if scene_index <= 0:
            return False, "first_scene"
        if scene_index - 1 >= len(scenes):
            return False, "previous_scene_missing"

        previous_scene = scenes[scene_index - 1]
        previous_cast = set(self._scene_cast_ids(previous_scene))
        current_cast = set(self._scene_cast_ids(scene))
        if previous_cast and current_cast and not (previous_cast & current_cast):
            return False, "cast_mismatch"
        return True, "continuity_anchor"

    def _ensure_continuity_input_available(
        self,
        scene_index: int,
        scene_id: str,
        scene: Dict[str, Any],
        continuity_frame_path: Optional[str],
        require_continuity: Optional[bool] = None,
    ) -> None:
        should_require = (
            self._scene_requires_continuity(scene_index=scene_index, scene=scene)
            if require_continuity is None
            else bool(require_continuity)
        )
        if not should_require:
            return
        candidate = str(continuity_frame_path or "").strip()
        if not candidate:
            raise RuntimeError(
                f"Continuity chain stopped before {scene_id}: missing previous scene end frame."
            )
        if not os.path.isfile(candidate):
            raise RuntimeError(
                f"Continuity chain stopped before {scene_id}: previous end frame not found at {candidate}."
            )

    def _ensure_last_frame_anchor_ready(
        self,
        *,
        scene_index: int,
        scene_id: str,
        scene: Dict[str, Any],
        continuity_frame_path: Optional[str],
        last_frame_image: Optional[types.Image],
        require_continuity: Optional[bool] = None,
    ) -> None:
        should_require = (
            self._scene_requires_continuity(scene_index=scene_index, scene=scene)
            if require_continuity is None
            else bool(require_continuity)
        )
        if not should_require:
            return
        if last_frame_image is not None:
            return
        candidate = str(continuity_frame_path or "").strip() or "<missing>"
        raise RuntimeError(
            f"Continuity chain stopped before {scene_id}: previous end frame exists but could not be loaded as a last-frame anchor ({candidate})."
        )

    @staticmethod
    def _scene_requires_continuity(scene_index: int, scene: Dict[str, Any]) -> bool:
        continuity = scene.get("continuity", {})
        if isinstance(continuity, dict) and continuity.get("must_follow_previous_frame") is not None:
            return bool(continuity.get("must_follow_previous_frame"))
        return scene_index > 0

    def _resolve_generation_aspect_ratio(
        self,
        manifest: Dict[str, Any],
        has_reference_images: bool,
        continuity_frame_path: Optional[str] = None,
    ) -> str:
        del has_reference_images
        continuity_ratio = self._infer_aspect_ratio_from_frame(continuity_frame_path)
        if continuity_ratio:
            return continuity_ratio
        output = manifest.get("output", {})
        aspect_ratio = str(output.get("aspect_ratio", "") or "").strip()
        if aspect_ratio in {"16:9", "9:16", "1:1"}:
            return aspect_ratio
        return "16:9"

    @staticmethod
    def _infer_aspect_ratio_from_frame(frame_path: Optional[str]) -> Optional[str]:
        path = str(frame_path or "").strip()
        if not path or not os.path.isfile(path):
            return None
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            return None
        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            return None
        ratio = width / float(height)
        if ratio >= 1.2:
            return "16:9"
        if ratio <= 0.9:
            return "9:16"
        return "1:1"

    def _build_last_frame_image(
        self,
        reference_inputs: List[Dict[str, str]],
    ) -> Optional[types.Image]:
        for item in reference_inputs:
            role = str(item.get("role", "")).strip().lower()
            if role != "continuity":
                continue
            path = str(item.get("path", "")).strip()
            if not path:
                continue
            try:
                return self._build_image_from_path(path)
            except Exception:
                continue
        return None

    def _build_image_from_path(self, path: str) -> types.Image:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is not None:
            max_dimension = 1536
            height, width = image.shape[:2]
            longest = max(height, width)
            if longest > max_dimension:
                scale = max_dimension / float(longest)
                resized = cv2.resize(
                    image,
                    (
                        max(1, int(width * scale)),
                        max(1, int(height * scale)),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
                image = resized

            encoded_ok, encoded = cv2.imencode(
                ".jpg",
                image,
                [int(cv2.IMWRITE_JPEG_QUALITY), 92],
            )
            if encoded_ok:
                return types.Image(
                    image_bytes=encoded.tobytes(),
                    mime_type="image/jpeg",
                )

            encoded_ok, encoded = cv2.imencode(".png", image)
            if encoded_ok:
                return types.Image(
                    image_bytes=encoded.tobytes(),
                    mime_type="image/png",
                )

        with open(path, "rb") as file:
            image_bytes = file.read()
        return types.Image(
            image_bytes=image_bytes,
            mime_type=self._guess_image_mime_type(path),
        )

    @staticmethod
    def _guess_image_mime_type(path: str) -> str:
        extension = os.path.splitext(path)[1].lower()
        if extension in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if extension == ".png":
            return "image/png"
        if extension == ".webp":
            return "image/webp"
        return "application/octet-stream"

    @staticmethod
    def _to_project_relative_path(path: str, project_dir: str) -> str:
        normalized = str(path or "").replace("\\", "/")
        if not normalized:
            return ""
        if not project_dir:
            return normalized
        try:
            rel = os.path.relpath(path, project_dir)
            return rel.replace("\\", "/")
        except Exception:
            return normalized

    def _build_subtitle_entries(self, manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
        entries = []
        for scene in manifest.get("scenes", []):
            for dialogue in scene.get("dialogue", []):
                entries.append(
                    {
                        "start_second": float(dialogue.get("start_second", 0)),
                        "end_second": float(dialogue.get("end_second", 0)),
                        "text": dialogue.get("line", ""),
                    }
                )
        entries.sort(key=lambda item: item["start_second"])
        return entries

    def _seed_continuity_from_previous_scene(
        self,
        *,
        manifest: Dict[str, Any],
        project_id: str,
        target_scene_id: str,
        project_dir: str,
        video_dir: str,
        frames_dir: str,
        warnings: List[Dict[str, Any]],
    ) -> Optional[str]:
        scenes = manifest.get("scenes", [])
        target_index = -1
        for idx, scene in enumerate(scenes):
            if isinstance(scene, dict) and scene.get("scene_id") == target_scene_id:
                target_index = idx
                break
        if target_index <= 0:
            return None

        previous_scene = scenes[target_index - 1]
        previous_scene_id = str(previous_scene.get("scene_id", "")).strip()
        if not previous_scene_id:
            return None

        debug = previous_scene.get("render_debug", {}) if isinstance(previous_scene, dict) else {}
        debug_frame_path = str(debug.get("continuity_frame_path", "")).strip()
        if debug_frame_path:
            candidate = debug_frame_path
            if not os.path.isabs(candidate):
                candidate = os.path.join(project_dir, candidate.replace("/", os.sep))
            if os.path.isfile(candidate):
                return candidate

        frame_prefix = f"{project_id}_{previous_scene_id}_"
        frame_candidates = [
            os.path.join(frames_dir, name)
            for name in os.listdir(frames_dir)
            if name.startswith(frame_prefix) and name.lower().endswith("_end.jpg")
        ]
        frame_candidates = [path for path in frame_candidates if os.path.isfile(path)]
        if frame_candidates:
            frame_candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
            return frame_candidates[0]

        previous_clip_path = ""
        debug_path = str(debug.get("output_clip_path", "")).strip()
        if debug_path:
            candidate = debug_path
            if not os.path.isabs(candidate):
                candidate = os.path.join(project_dir, candidate.replace("/", os.sep))
            if os.path.isfile(candidate):
                previous_clip_path = candidate

        if not previous_clip_path:
            candidates = [
                os.path.join(video_dir, name)
                for name in os.listdir(video_dir)
                if name.startswith(frame_prefix) and name.lower().endswith(".mp4")
            ]
            candidates = [path for path in candidates if os.path.isfile(path)]
            if candidates:
                candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
                previous_clip_path = candidates[0]

        if not previous_clip_path:
            warnings.append(
                {
                    "category": "continuity_seed_missing",
                    "severity": "warning",
                    "message": (
                        f"Could not locate previous scene clip for continuity seed before {target_scene_id}."
                    ),
                    "field_path": f"scenes.{target_scene_id}",
                }
            )
            return None

        continuity_seed_path = os.path.join(
            frames_dir,
            f"{project_id}_{target_scene_id}_continuity_seed.jpg",
        )
        try:
            self._extract_last_frame(previous_clip_path, continuity_seed_path)
            return continuity_seed_path
        except Exception as error:
            warnings.append(
                {
                    "category": "continuity_seed_extract_failure",
                    "severity": "warning",
                    "message": (
                        f"Could not extract continuity seed frame for {target_scene_id}: {error}"
                    ),
                    "field_path": f"scenes.{target_scene_id}",
                }
            )
            return None

    def _extract_last_frame(self, video_path: str, output_frame_path: str) -> None:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise IOError(f"Cannot open video {video_path}")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            capture.release()
            raise IOError("Video has no frames.")
        capture.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise IOError("Failed to read last frame.")
        os.makedirs(os.path.dirname(output_frame_path), exist_ok=True)
        saved = cv2.imwrite(output_frame_path, frame)
        if not saved:
            raise IOError(f"Failed to write frame to {output_frame_path}")

    def _create_placeholder_clip(
        self,
        output_path: str,
        scene_id: str,
        scene: Dict[str, Any],
    ) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        width, height = 720, 1280
        fps = 24
        seconds = max(
            2,
            int(
                sum(
                    float(shot.get("duration_seconds", 0))
                    for shot in scene.get("shot_list", [])
                )
            ),
        )

        writer = None
        for codec in ("mp4v", "MJPG", "avc1", "H264", "X264"):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            candidate = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if candidate.isOpened():
                writer = candidate
                break
            candidate.release()
        if writer is None or not writer.isOpened():
            raise RuntimeError("Could not open video writer for placeholder clip.")

        text1 = f"{scene_id} placeholder"
        text2 = scene.get("scene_goal", "Auto-generated scene")
        for _ in range(seconds * fps):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = (24, 24, 24)
            cv2.putText(
                frame,
                text1[:48],
                (40, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                text2[:62],
                (40, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (180, 220, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)

        writer.release()

    def _stitch_final_video(
        self,
        clip_records: List[Dict[str, str]],
        voice_tracks: List[Dict[str, Any]],
        subtitle_path: str,
        output_path: str,
        output_spec: Dict[str, Any],
        timeline: Dict[str, Any],
        warnings: List[Dict[str, Any]],
    ) -> str:
        clip_paths = [item["clip_path"] for item in clip_records if item.get("clip_path")]
        if not clip_paths:
            raise RuntimeError("No clips produced for final render.")

        ffmpeg_path = shutil.which("ffmpeg")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if not ffmpeg_path:
            shutil.copyfile(clip_paths[0], output_path)
            warnings.append(
                {
                    "category": "ffmpeg_missing",
                    "severity": "warning",
                    "message": "FFmpeg not found. Final output is first clip only.",
                    "field_path": "output",
                }
            )
            return output_path

        trimmed_clips = self._apply_scene_trims(
            ffmpeg_path=ffmpeg_path,
            clip_records=clip_records,
            timeline=timeline,
            warnings=warnings,
        )
        paced_clips = self._apply_scene_pacing(
            ffmpeg_path=ffmpeg_path,
            clip_records=trimmed_clips,
            timeline=timeline,
            warnings=warnings,
        )

        transitions = timeline.get("transitions", []) if isinstance(timeline, dict) else []
        temp_base = output_path.replace(".mp4", "_base.mp4")
        if self._has_non_cut_transition(transitions):
            stitched = self._stitch_with_transitions(
                ffmpeg_path=ffmpeg_path,
                clip_records=paced_clips,
                transitions=transitions,
                output_path=temp_base,
                fps=str(output_spec.get("fps", 30)),
                warnings=warnings,
            )
            if not stitched:
                temp_base = self._concat_clips(
                    ffmpeg_path,
                    [item["clip_path"] for item in paced_clips],
                    output_path,
                    output_spec,
                    warnings,
                )
        else:
            temp_base = self._concat_clips(
                ffmpeg_path,
                [item["clip_path"] for item in paced_clips],
                output_path,
                output_spec,
                warnings,
            )

        temp_overlay = self._apply_b_roll(
            ffmpeg_path=ffmpeg_path,
            base_video_path=temp_base,
            timeline=timeline,
            output_path=output_path.replace(".mp4", "_broll.mp4"),
            warnings=warnings,
        )
        current_video = temp_overlay or temp_base

        if voice_tracks:
            mixed_path = output_path.replace(".mp4", "_mixed.mp4")
            mix_cmd = self._build_audio_mix_command(
                ffmpeg_path,
                video_path=current_video,
                voice_tracks=voice_tracks,
                output_path=mixed_path,
            )
            try:
                subprocess.run(mix_cmd, check=True, capture_output=True)
                current_video = mixed_path
            except subprocess.CalledProcessError:
                warnings.append(
                    {
                        "category": "audio_mix_failure",
                        "severity": "warning",
                        "message": "Audio mix failed; continuing with video-only track.",
                        "field_path": "audio",
                    }
                )

        if subtitle_path and os.path.exists(subtitle_path) and output_spec.get(
            "subtitles_enabled", True
        ):
            burned_path = output_path.replace(".mp4", "_subtitled.mp4")
            subtitle_filter = self._ffmpeg_subtitle_filter(subtitle_path)
            subtitle_cmd = [
                ffmpeg_path,
                "-y",
                "-i",
                current_video,
                "-vf",
                subtitle_filter,
                "-c:v",
                "libx264",
                "-c:a",
                "copy",
                burned_path,
            ]
            try:
                subprocess.run(subtitle_cmd, check=True, capture_output=True)
                current_video = burned_path
            except subprocess.CalledProcessError:
                warnings.append(
                    {
                        "category": "subtitle_burn_failure",
                        "severity": "warning",
                        "message": "Subtitle burn-in failed; exporting without burnt subtitles.",
                        "field_path": "output.subtitles",
                    }
                )

        shutil.copyfile(current_video, output_path)
        return output_path

    def _apply_scene_trims(
        self,
        ffmpeg_path: str,
        clip_records: List[Dict[str, str]],
        timeline: Dict[str, Any],
        warnings: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        if not isinstance(timeline, dict):
            return clip_records
        trim_map = timeline.get("scene_trims", {})
        if not isinstance(trim_map, dict):
            return clip_records

        processed = []
        for record in clip_records:
            scene_id = record.get("scene_id")
            trim_data = trim_map.get(scene_id, {})
            if not isinstance(trim_data, dict):
                processed.append(record)
                continue
            start_trim = float(trim_data.get("start_trim", 0.0) or 0.0)
            end_trim = float(trim_data.get("end_trim", 0.0) or 0.0)
            if start_trim < 0 or end_trim < 0:
                start_trim = 0.0
                end_trim = 0.0
            if start_trim <= 0.001 and end_trim <= 0.001:
                processed.append(record)
                continue

            clip_path = record["clip_path"]
            duration = self._video_duration_seconds(clip_path)
            if duration <= 0.0:
                processed.append(record)
                continue
            keep_start = min(start_trim, max(0.0, duration - 0.2))
            keep_end = max(keep_start + 0.2, duration - end_trim)
            if keep_end <= keep_start:
                warnings.append(
                    {
                        "category": "trim_invalid_range",
                        "severity": "warning",
                        "message": f"Invalid trim for {scene_id}; using original clip.",
                        "field_path": f"timeline.scene_trims.{scene_id}",
                    }
                )
                processed.append(record)
                continue

            trimmed_path = clip_path.replace(".mp4", "_trimmed.mp4")
            cmd = [
                ffmpeg_path,
                "-y",
                "-i",
                clip_path,
                "-ss",
                str(round(keep_start, 3)),
                "-to",
                str(round(keep_end, 3)),
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                trimmed_path,
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                processed.append({**record, "clip_path": trimmed_path})
            except subprocess.CalledProcessError:
                warnings.append(
                    {
                        "category": "trim_failure",
                        "severity": "warning",
                        "message": f"Could not trim scene {scene_id}; using original clip.",
                        "field_path": f"timeline.scene_trims.{scene_id}",
                    }
                )
                processed.append(record)
        return processed

    def _ensure_web_playable_clip(
        self,
        clip_path: str,
        warnings: List[Dict[str, Any]],
    ) -> str:
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            return clip_path
        if not os.path.exists(clip_path):
            return clip_path

        normalized_path = clip_path.replace(".mp4", "_websafe.mp4")
        commands = [
            [
                ffmpeg_path,
                "-y",
                "-i",
                clip_path,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-c:a",
                "aac",
                normalized_path,
            ],
            [
                ffmpeg_path,
                "-y",
                "-i",
                clip_path,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                normalized_path,
            ],
        ]

        for command in commands:
            try:
                subprocess.run(command, check=True, capture_output=True)
                shutil.move(normalized_path, clip_path)
                return clip_path
            except subprocess.CalledProcessError:
                continue
            finally:
                if os.path.exists(normalized_path):
                    try:
                        os.remove(normalized_path)
                    except OSError:
                        pass

        warnings.append(
            {
                "category": "clip_web_playback_warning",
                "severity": "warning",
                "message": (
                    "Could not normalize clip to browser-safe H.264; playback may fail in some browsers."
                ),
                "field_path": "output",
            }
        )
        return clip_path

    def _apply_scene_pacing(
        self,
        ffmpeg_path: str,
        clip_records: List[Dict[str, str]],
        timeline: Dict[str, Any],
        warnings: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        pacing = timeline.get("pacing", {}) if isinstance(timeline, dict) else {}
        if not isinstance(pacing, dict):
            return clip_records

        processed = []
        for record in clip_records:
            scene_id = record.get("scene_id")
            speed = float(pacing.get(scene_id, 1.0) or 1.0)
            clip_path = record["clip_path"]
            if abs(speed - 1.0) < 0.01:
                processed.append(record)
                continue
            speed = max(0.5, min(2.0, speed))
            paced_path = clip_path.replace(".mp4", "_paced.mp4")
            cmd = [
                ffmpeg_path,
                "-y",
                "-i",
                clip_path,
                "-filter:v",
                f"setpts=PTS/{speed}",
                "-an",
                paced_path,
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                processed.append({**record, "clip_path": paced_path})
            except subprocess.CalledProcessError:
                warnings.append(
                    {
                        "category": "pacing_failure",
                        "severity": "warning",
                        "message": f"Could not apply pacing for {scene_id}; using original speed.",
                        "field_path": f"timeline.pacing.{scene_id}",
                    }
                )
                processed.append(record)
        return processed

    def _concat_clips(
        self,
        ffmpeg_path: str,
        clip_paths: List[str],
        output_path: str,
        output_spec: Dict[str, Any],
        warnings: List[Dict[str, Any]],
    ) -> str:
        temp_concat = output_path.replace(".mp4", "_concat.mp4")
        concat_list = output_path.replace(".mp4", "_concat.txt")
        with open(concat_list, "w", encoding="utf-8") as file:
            for clip in clip_paths:
                file.write(f"file '{clip.replace('\\', '/')}'\n")

        fps = str(output_spec.get("fps", 30))
        concat_cmd = [
            ffmpeg_path,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list,
            "-r",
            fps,
            "-pix_fmt",
            "yuv420p",
            temp_concat,
        ]
        try:
            subprocess.run(concat_cmd, check=True, capture_output=True)
            return temp_concat
        except subprocess.CalledProcessError:
            shutil.copyfile(clip_paths[0], temp_concat)
            warnings.append(
                {
                    "category": "concat_failure",
                    "severity": "warning",
                    "message": "Clip concat failed. Base output is first clip only.",
                    "field_path": "output",
                }
            )
            return temp_concat

    def _stitch_with_transitions(
        self,
        ffmpeg_path: str,
        clip_records: List[Dict[str, str]],
        transitions: List[Dict[str, Any]],
        output_path: str,
        fps: str,
        warnings: List[Dict[str, Any]],
    ) -> bool:
        if len(clip_records) < 2:
            return False
        clips = [item["clip_path"] for item in clip_records]
        cmd = [ffmpeg_path, "-y"]
        for clip in clips:
            cmd += ["-i", clip]

        durations = [self._video_duration_seconds(path) for path in clips]
        filter_parts = []
        current_label = "[0:v]"
        elapsed = durations[0]
        for idx in range(1, len(clips)):
            from_scene = clip_records[idx - 1].get("scene_id")
            to_scene = clip_records[idx].get("scene_id")
            transition = self._transition_for_pair(transitions, from_scene, to_scene)
            kind = str(transition.get("type", "fade")).lower()
            duration = float(transition.get("duration_seconds", 0.4) or 0.4)
            if kind == "cut":
                kind = "fade"
                duration = 0.05
            duration = max(0.05, min(2.0, duration))
            offset = max(0.0, elapsed - duration)
            out_label = f"[v{idx}]"
            filter_parts.append(
                f"{current_label}[{idx}:v]xfade=transition={kind}:duration={duration}:offset={offset}{out_label}"
            )
            current_label = out_label
            elapsed = offset + durations[idx]

        filter_complex = ";".join(filter_parts)
        cmd += [
            "-filter_complex",
            filter_complex,
            "-map",
            current_label,
            "-r",
            fps,
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            warnings.append(
                {
                    "category": "transition_failure",
                    "severity": "warning",
                    "message": "Transition render failed; falling back to hard concat.",
                    "field_path": "timeline.transitions",
                }
            )
            return False

    def _apply_b_roll(
        self,
        ffmpeg_path: str,
        base_video_path: str,
        timeline: Dict[str, Any],
        output_path: str,
        warnings: List[Dict[str, Any]],
    ) -> str:
        b_roll = timeline.get("b_roll", []) if isinstance(timeline, dict) else []
        if not isinstance(b_roll, list) or not b_roll:
            return ""

        valid = []
        for item in b_roll:
            path = item.get("path")
            if isinstance(path, str) and os.path.isfile(path):
                valid.append(item)
        if not valid:
            return ""

        cmd = [ffmpeg_path, "-y", "-i", base_video_path]
        for item in valid:
            cmd += ["-stream_loop", "-1", "-i", item["path"]]

        filter_parts = []
        previous = "[0:v]"
        for idx, item in enumerate(valid, start=1):
            start = float(item.get("start_second", 0.0))
            duration = float(item.get("duration_seconds", 2.0))
            end = start + max(0.2, duration)
            overlay_label = f"[ov{idx}]"
            filter_parts.append(
                f"{previous}[{idx}:v]overlay=0:0:enable='between(t,{start},{end})'{overlay_label}"
            )
            previous = overlay_label

        cmd += [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            previous,
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-c:a",
            "copy",
            output_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError:
            warnings.append(
                {
                    "category": "broll_overlay_failure",
                    "severity": "warning",
                    "message": "B-roll overlay failed; continuing without B-roll.",
                    "field_path": "timeline.b_roll",
                }
            )
            return ""

    def _video_duration_seconds(self, path: str) -> float:
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            return 0.0
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
        capture.release()
        if fps <= 0:
            fps = 24.0
        return max(0.0, frames / fps)

    def _has_non_cut_transition(self, transitions: List[Dict[str, Any]]) -> bool:
        for transition in transitions:
            if str(transition.get("type", "cut")).lower() != "cut":
                return True
        return False

    def _transition_for_pair(
        self,
        transitions: List[Dict[str, Any]],
        from_scene: Optional[str],
        to_scene: Optional[str],
    ) -> Dict[str, Any]:
        for transition in transitions:
            if (
                transition.get("from_scene_id") == from_scene
                and transition.get("to_scene_id") == to_scene
            ):
                return transition
        return {"type": "cut", "duration_seconds": 0.0}

    def _build_audio_mix_command(
        self,
        ffmpeg_path: str,
        video_path: str,
        voice_tracks: List[Dict[str, Any]],
        output_path: str,
    ) -> List[str]:
        cmd = [ffmpeg_path, "-y", "-i", video_path]
        valid_tracks = [track for track in voice_tracks if os.path.exists(track["path"])]
        if not valid_tracks:
            return [ffmpeg_path, "-y", "-i", video_path, "-c", "copy", output_path]

        for track in valid_tracks:
            cmd += ["-i", track["path"]]

        filter_parts = []
        mix_labels = []
        for idx, track in enumerate(valid_tracks, start=1):
            delay_ms = int(float(track.get("start_second", 0)) * 1000)
            label = f"a{idx}"
            filter_parts.append(f"[{idx}:a]adelay={delay_ms}|{delay_ms}[{label}]")
            mix_labels.append(f"[{label}]")

        mix_inputs = "".join(mix_labels)
        filter_parts.append(f"{mix_inputs}amix=inputs={len(valid_tracks)}:normalize=0[aout]")
        filter_complex = ";".join(filter_parts)

        cmd += [
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            output_path,
        ]
        return cmd

    def _ffmpeg_subtitle_filter(self, subtitle_path: str) -> str:
        escaped = subtitle_path.replace("\\", "/").replace(":", "\\:")
        return f"subtitles='{escaped}'"

    def _write_reports_and_artifacts(
        self,
        manifest: Dict[str, Any],
        paths: Dict[str, str],
        final_video_path: str,
        clip_records: List[Dict[str, str]],
        voice_tracks: List[Dict[str, Any]],
        subtitle_path: str,
        warnings: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        reports_dir = paths["reports_dir"]
        script_path = os.path.join(reports_dir, "script.txt")
        scene_list_path = os.path.join(reports_dir, "scene_list.json")
        assets_list_path = os.path.join(reports_dir, "assets_list.json")
        autofill_report_path = os.path.join(reports_dir, "autofill_report.txt")
        safety_report_path = os.path.join(reports_dir, "safety_report.txt")
        audit_report_path = os.path.join(reports_dir, "audit_report.txt")
        scene_builder_plan_path = os.path.join(reports_dir, "scene_builder_plan.json")
        scene_render_debug_path = os.path.join(reports_dir, "scene_render_debug.json")
        scene_builder_narrative_path = os.path.join(
            reports_dir, "scene_builder_narrative.txt"
        )

        self.report_service.write_script_report(manifest, script_path)
        self.report_service.write_scene_list(manifest, scene_list_path)
        self.report_service.write_autofill_report(manifest, autofill_report_path)
        self.report_service.write_safety_report(warnings, safety_report_path)
        self.report_service.write_scene_builder_plan(manifest, scene_builder_plan_path)
        self.report_service.write_scene_render_debug(manifest, scene_render_debug_path)
        self.report_service.write_scene_builder_narrative(
            manifest, scene_builder_narrative_path
        )
        self.report_service.write_audit_report(
            manifest=manifest,
            warnings=warnings,
            voice_tracks=voice_tracks,
            output_path=audit_report_path,
        )

        assets: List[Dict[str, Any]] = [
            {"type": "final_video", "path": final_video_path.replace("\\", "/")},
            {"type": "manifest", "path": paths["manifest"].replace("\\", "/")},
            {"type": "autofill_log", "path": paths["autofill_log"].replace("\\", "/")},
            {"type": "script", "path": script_path.replace("\\", "/")},
            {"type": "scene_list", "path": scene_list_path.replace("\\", "/")},
            {"type": "autofill_report", "path": autofill_report_path.replace("\\", "/")},
            {"type": "safety_report", "path": safety_report_path.replace("\\", "/")},
            {"type": "scene_builder_plan", "path": scene_builder_plan_path.replace("\\", "/")},
            {"type": "scene_render_debug", "path": scene_render_debug_path.replace("\\", "/")},
            {
                "type": "scene_builder_narrative",
                "path": scene_builder_narrative_path.replace("\\", "/"),
            },
            {"type": "audit_report", "path": audit_report_path.replace("\\", "/")},
        ]
        for clip in clip_records:
            assets.append({"type": "clip", "path": clip.get("clip_path", "").replace("\\", "/")})
            if clip.get("end_frame_path"):
                assets.append(
                    {
                        "type": "continuity_frame",
                        "path": clip.get("end_frame_path", "").replace("\\", "/"),
                    }
                )
        for track in voice_tracks:
            assets.append({"type": "audio_track", "path": track["path"].replace("\\", "/")})
        if subtitle_path and os.path.exists(subtitle_path):
            assets.append({"type": "subtitle", "path": subtitle_path.replace("\\", "/")})

        self.report_service.write_assets_list(assets, assets_list_path)
        self.report_service.write_assets_list(
            manifest.get("autofill_log", []), paths["autofill_log"]
        )
        return assets
