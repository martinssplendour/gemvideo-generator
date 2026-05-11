from __future__ import annotations

from typing import Any, Dict, List

from schemas import REQUIRED_BRIEF_FIELDS


class ValidationError(ValueError):
    pass


class ValidationService:
    def normalize_manifest(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        brief = manifest.setdefault("brief", {})
        if brief.get("aspect_ratio"):
            brief["aspect_ratio"] = self._normalize_aspect_ratio(brief["aspect_ratio"])

        if brief.get("platform"):
            brief["platform"] = self._normalize_platform(brief["platform"])

        if brief.get("duration_seconds") is not None:
            try:
                brief["duration_seconds"] = int(brief["duration_seconds"])
            except (TypeError, ValueError):
                brief["duration_seconds"] = None

        if brief.get("scene_count_hint") is not None:
            try:
                brief["scene_count_hint"] = max(0, int(brief["scene_count_hint"]))
            except (TypeError, ValueError):
                brief["scene_count_hint"] = 0

        output = manifest.setdefault("output", {})
        if output.get("fps") is not None:
            try:
                output["fps"] = int(output["fps"])
            except (TypeError, ValueError):
                output["fps"] = 30

        output.setdefault("file_format", "mp4")
        output.setdefault("resolution", "1080p")
        output.setdefault("subtitles_enabled", True)
        output.setdefault("subtitle_style", "clean lower-third")
        output.setdefault("safe_areas", "social-safe margins")
        output.setdefault(
            "deliverables", ["final_video", "script", "scene_list", "assets_list"]
        )

        manifest.setdefault("characters", [])
        manifest.setdefault("world", {})
        manifest.setdefault("scenes", [])
        manifest.setdefault("generation_mode", "auto")
        if manifest.get("generation_mode") not in ("auto", "directed"):
            manifest["generation_mode"] = "auto"
        timeline = manifest.get("timeline")
        if isinstance(timeline, list):
            manifest["timeline"] = {
                "entries": timeline,
                "scene_order": [],
                "transitions": [],
                "pacing": {},
                "b_roll": [],
                "scene_trims": {},
            }
        elif not isinstance(timeline, dict):
            manifest["timeline"] = {
                "entries": [],
                "scene_order": [],
                "transitions": [],
                "pacing": {},
                "b_roll": [],
                "scene_trims": {},
            }
        else:
            timeline.setdefault("entries", [])
            timeline.setdefault("scene_order", [])
            timeline.setdefault("transitions", [])
            timeline.setdefault("pacing", {})
            timeline.setdefault("b_roll", [])
            timeline.setdefault("scene_trims", {})
        manifest.setdefault("audio", {"music_mood": "", "sound_effects": []})
        manifest.setdefault("autofill_log", [])
        manifest.setdefault("safety_warnings", [])
        manifest.setdefault("surprise_counters", {})
        manifest.setdefault("scene_variants", {})
        manifest.setdefault("character_bible", {})
        manifest.setdefault(
            "approvals",
            {
                "script_approved": False,
                "script_needs_user_approval": True,
                "script_last_reviewed_at": None,
            },
        )
        manifest.setdefault("render_iteration", 0)

        for scene in manifest.get("scenes", []):
            if not isinstance(scene, dict):
                continue
            scene.setdefault("locked", False)
            scene["scene_type"] = self._normalize_scene_type(scene.get("scene_type"))
            scene_state = scene.get("state")
            if scene_state not in {"planned", "generated", "approved", "locked", "stitched"}:
                scene_state = "locked" if scene.get("locked") else "planned"
            scene["state"] = scene_state

        return manifest

    def validate_brief_required(self, manifest: Dict[str, Any]) -> List[str]:
        brief = manifest.get("brief", {})
        missing: List[str] = []
        for field in REQUIRED_BRIEF_FIELDS:
            value = brief.get(field)
            if value in (None, "", []):
                missing.append(field)
        return missing

    def ensure_render_eligible(self, manifest: Dict[str, Any]) -> None:
        missing = self.validate_brief_required(manifest)
        if missing:
            raise ValidationError(
                "Missing required brief fields: " + ", ".join(sorted(missing))
            )

        if not manifest.get("scenes"):
            raise ValidationError("Manifest has no scenes after auto-fill.")

        approvals = manifest.get("approvals", {})
        if not approvals.get("script_approved"):
            raise ValidationError(
                "Script must be approved before render. Set approvals.script_approved=true."
            )

        for scene in manifest.get("scenes", []):
            if not scene.get("dialogue"):
                raise ValidationError(
                    f"Scene {scene.get('scene_id', 'unknown')} is missing dialogue."
                )
            if not scene.get("shot_list"):
                raise ValidationError(
                    f"Scene {scene.get('scene_id', 'unknown')} is missing shot list."
                )

    @staticmethod
    def _normalize_aspect_ratio(value: Any) -> str:
        text = str(value or "").strip().lower()
        if "9:16" in text or "vertical" in text or "portrait" in text:
            return "9:16"
        if "16:9" in text or "landscape" in text:
            return "16:9"
        if "1:1" in text or "square" in text:
            return "1:1"
        return str(value or "").strip()

    @staticmethod
    def _normalize_platform(value: Any) -> str:
        text = str(value or "").strip().lower()
        if "tiktok" in text:
            return "TikTok"
        if "instagram" in text or "reels" in text:
            return "Instagram"
        if "youtube" in text:
            return "YouTube"
        if "linkedin" in text:
            return "LinkedIn"
        if "website" in text:
            return "Website"
        return str(value or "").strip()

    @staticmethod
    def _normalize_scene_type(value: Any) -> str:
        text = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
        if not text:
            return "auto"
        allowed = {
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
        return text if text in allowed else "custom"
