from __future__ import annotations

import json
import os
from typing import Any, Dict, List


class ReportService:
    def write_autofill_report(
        self,
        manifest: Dict[str, Any],
        report_path: str,
    ) -> None:
        lines = [
            "Auto-fill Report",
            "===============",
            f"Project ID: {manifest.get('project_id', '')}",
            f"Seed: {manifest.get('project_seed', '')}",
            "",
        ]
        for item in manifest.get("autofill_log", []):
            lines.append(
                f"- {item.get('field_path')}: {item.get('reason')} "
                f"[seed={item.get('seed_fragment')} at {item.get('timestamp')}]"
            )
        self._write_text(report_path, "\n".join(lines))

    def write_safety_report(
        self,
        warnings: List[Dict[str, Any]],
        report_path: str,
    ) -> None:
        lines = ["Safety Report", "============", ""]
        if not warnings:
            lines.append("No warnings detected.")
        else:
            for warning in warnings:
                lines.append(
                    f"- [{warning.get('category')}] {warning.get('message')} "
                    f"({warning.get('field_path')})"
                )
        self._write_text(report_path, "\n".join(lines))

    def write_script_report(self, manifest: Dict[str, Any], report_path: str) -> None:
        lines = ["Script", "======", ""]
        for scene in manifest.get("scenes", []):
            lines.append(f"{scene.get('scene_id')}: {scene.get('scene_goal', '')}")
            for line in scene.get("dialogue", []):
                speaker = line.get("speaker_id", "speaker")
                text = line.get("line", "")
                lines.append(f"  {speaker}: {text}")
            lines.append("")
        self._write_text(report_path, "\n".join(lines))

    def write_scene_list(self, manifest: Dict[str, Any], output_path: str) -> None:
        scenes = []
        for scene in manifest.get("scenes", []):
            scenes.append(
                {
                    "scene_id": scene.get("scene_id"),
                    "scene_goal": scene.get("scene_goal"),
                    "scene_description": scene.get("scene_description", ""),
                    "spoken_direction": scene.get("spoken_direction", ""),
                    "image_usage_instructions": scene.get("image_usage_instructions", ""),
                    "scene_type": scene.get("scene_type", "auto"),
                    "location_time": scene.get("location_time"),
                    "events": scene.get("events", []),
                    "characters_in_scene": scene.get("characters_in_scene", []),
                    "locked": scene.get("locked", False),
                }
            )
        self._write_json(output_path, scenes)

    def write_assets_list(self, assets: List[Dict[str, Any]], output_path: str) -> None:
        self._write_json(output_path, assets)

    def write_scene_builder_plan(self, manifest: Dict[str, Any], output_path: str) -> None:
        payload: List[Dict[str, Any]] = []
        for scene in manifest.get("scenes", []):
            if not isinstance(scene, dict):
                continue
            payload.append(
                {
                    "scene_id": scene.get("scene_id"),
                    "scene_type": scene.get("scene_type"),
                    "scene_goal": scene.get("scene_goal"),
                    "scene_description": scene.get("scene_description", ""),
                    "spoken_direction": scene.get("spoken_direction", ""),
                    "image_usage_instructions": scene.get("image_usage_instructions", ""),
                    "location_time": scene.get("location_time", ""),
                    "characters_in_scene": scene.get("characters_in_scene", []),
                    "events": scene.get("events", []),
                    "continuity": scene.get("continuity", {}),
                    "builder": scene.get("builder", {}),
                }
            )
        self._write_json(output_path, payload)

    def write_scene_render_debug(self, manifest: Dict[str, Any], output_path: str) -> None:
        payload: List[Dict[str, Any]] = []
        for scene in manifest.get("scenes", []):
            if not isinstance(scene, dict):
                continue
            debug = scene.get("render_debug", {})
            payload.append(
                {
                    "scene_id": scene.get("scene_id"),
                    "generation_status": debug.get("generation_status", ""),
                    "veo_prompt": debug.get("veo_prompt", ""),
                    "reference_images": debug.get("reference_images", []),
                    "output_clip_path": debug.get("output_clip_path", ""),
                    "scene_builder_snapshot": debug.get("scene_builder_snapshot", {}),
                }
            )
        self._write_json(output_path, payload)

    def write_scene_builder_narrative(self, manifest: Dict[str, Any], output_path: str) -> None:
        lines: List[str] = ["Scene Builder Narrative", "=======================", ""]
        for scene in manifest.get("scenes", []):
            if not isinstance(scene, dict):
                continue
            lines.append(f"{scene.get('scene_id', 'scene')}: {scene.get('scene_goal', '')}")
            lines.append(f"Description: {scene.get('scene_description', '')}")
            lines.append(f"Spoken Direction: {scene.get('spoken_direction', '')}")
            lines.append(
                f"Image Usage: {scene.get('image_usage_instructions', '')}"
            )
            lines.append("")
        self._write_text(output_path, "\n".join(lines))

    def write_audit_report(
        self,
        manifest: Dict[str, Any],
        warnings: List[Dict[str, Any]],
        voice_tracks: List[Dict[str, Any]],
        output_path: str,
    ) -> None:
        lines = [
            "Audit Report",
            "============",
            f"Project ID: {manifest.get('project_id', '')}",
            f"Seed: {manifest.get('project_seed', '')}",
            f"Generation mode: {manifest.get('generation_mode', 'auto')}",
            f"Render iteration: {manifest.get('render_iteration', 0)}",
            "",
        ]

        lines.append("Scene states:")
        for scene in manifest.get("scenes", []):
            lines.append(
                f"- {scene.get('scene_id', 'scene')}: state={scene.get('state', 'planned')} locked={scene.get('locked', False)}"
            )
        lines.append("")

        lines.append(f"Safety warnings: {len(warnings)}")
        for warning in warnings:
            lines.append(
                f"- [{warning.get('category', 'unknown')}] {warning.get('message', '')}"
            )
        lines.append("")

        lines.append("Voice QA:")
        fallback_count = 0
        for track in voice_tracks:
            fallback_reason = track.get("fallback_reason", "")
            if fallback_reason:
                fallback_count += 1
            qa = track.get("qa", {})
            lines.append(
                f"- {track.get('scene_id')}::{track.get('speaker_id')} "
                f"provider={track.get('voice_provider')} "
                f"duration_fit={qa.get('duration_fit', True)} "
                f"clipping={qa.get('clipping_detected', False)} "
                f"fallback={fallback_reason or 'none'}"
            )
        lines.append(f"Voice fallback count: {fallback_count}")

        self._write_text(output_path, "\n".join(lines))

    def write_subtitle_file(self, subtitle_entries: List[Dict[str, Any]], output_path: str) -> None:
        lines: List[str] = []
        for index, item in enumerate(subtitle_entries, start=1):
            lines.append(str(index))
            lines.append(
                f"{self._format_ts(item['start_second'])} --> {self._format_ts(item['end_second'])}"
            )
            lines.append(item["text"])
            lines.append("")
        self._write_text(output_path, "\n".join(lines))

    def _write_text(self, path: str, content: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            file.write(content)

    def _write_json(self, path: str, data: Any) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _format_ts(seconds: float) -> str:
        milliseconds = int(round(seconds * 1000))
        hours, rem = divmod(milliseconds, 3_600_000)
        minutes, rem = divmod(rem, 60_000)
        secs, ms = divmod(rem, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
