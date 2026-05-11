from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types


class DescriptionPlannerService:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        timeout_seconds: int = 12,
        enabled: bool = True,
        client: Optional[Any] = None,
    ):
        key = str(api_key or "").strip()
        self.model_name = model_name
        self.timeout_seconds = int(timeout_seconds)
        self.enabled = bool(enabled) and bool(key) and key.lower() not in {
            "dummy_key",
            "test_key",
            "placeholder",
        }
        self.client = client
        if self.enabled and self.client is None:
            self.client = genai.Client(api_key=key)

    def plan_manifest_patch(
        self,
        description: str,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        text = str(description or "").strip()
        if not self.enabled or not text or self.client is None:
            return {}

        payload_context = context if isinstance(context, dict) else {}
        prompt = self._build_prompt(text, payload_context)
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )
        except Exception:
            return {}

        raw_text = self._extract_text(response)
        if not raw_text:
            return {}
        parsed = self._parse_json_like(raw_text)
        if not isinstance(parsed, dict):
            return {}
        return self._sanitize_plan(parsed)

    def _build_prompt(self, description: str, context: Dict[str, Any]) -> str:
        context_payload = {
            "brief": context.get("brief", {}),
            "world": context.get("world", {}),
            "characters": context.get("characters", []),
            "scenes_count": len(context.get("scenes", []))
            if isinstance(context.get("scenes"), list)
            else 0,
        }
        return (
            "You are planning a short-form video manifest.\n"
            "Given the user description and current manifest context, infer the best assumptions.\n"
            "Return strict JSON object only with keys:\n"
            "- brief (object): any of video_type, goal, target_audience, platform, duration_seconds, "
            "aspect_ratio, tone, language, accent, brand_rules, logline, scene_count_hint\n"
            "- characters (array of objects): name, role, appearance, personality, constraints, optional voice\n"
            "- world (object): setting, background_style, color_palette, wardrobe_props, camera_style, lighting, motion\n"
            "- scene_plan (array): each item has scene_goal and optional scene_type "
            "(hook/problem/reframe/solution/payoff/cta/b_roll/custom)\n"
            "- scene_plan_text (string): newline-separated scene goals (fallback)\n"
            "- script_text (string): simple script lines, can use 'Speaker: line'\n"
            "- audio (object): music_mood, sound_effects\n"
            "- output (object): resolution, fps, subtitles_enabled, subtitle_style, safe_areas, file_format\n"
            "- assumptions (array): short assumption notes\n"
            "Rules:\n"
            "1) Natural language values only.\n"
            "2) Prefer platform-fit pacing (TikTok/IG hooks in first 1-2 seconds).\n"
            "3) Avoid real-person impersonation.\n"
            "4) Keep output concise and production-usable.\n\n"
            f"Description:\n{description}\n\n"
            f"Current context JSON:\n{json.dumps(context_payload, ensure_ascii=False)}\n"
        )

    @staticmethod
    def _extract_text(response: Any) -> str:
        text = getattr(response, "text", "")
        if isinstance(text, str) and text.strip():
            return text.strip()

        candidates = getattr(response, "candidates", None)
        if isinstance(candidates, list):
            fragments: List[str] = []
            for candidate in candidates:
                content = getattr(candidate, "content", None)
                parts = getattr(content, "parts", None) if content is not None else None
                if not isinstance(parts, list):
                    continue
                for part in parts:
                    part_text = getattr(part, "text", "")
                    if isinstance(part_text, str) and part_text.strip():
                        fragments.append(part_text.strip())
            if fragments:
                return "\n".join(fragments)
        return ""

    @staticmethod
    def _parse_json_like(raw_text: str) -> Any:
        content = raw_text.strip()
        if content.startswith("```"):
            content = content.strip("`")
            if "\n" in content:
                content = content.split("\n", 1)[1]
        content = content.strip()
        if content.endswith("```"):
            content = content[: -3].strip()

        try:
            return json.loads(content)
        except Exception:
            pass

        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except Exception:
                return {}
        return {}

    def _sanitize_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}

        brief = plan.get("brief")
        if isinstance(brief, dict):
            cleaned_brief: Dict[str, Any] = {}
            allowed_brief_keys = {
                "video_type",
                "goal",
                "target_audience",
                "platform",
                "duration_seconds",
                "aspect_ratio",
                "tone",
                "language",
                "accent",
                "brand_rules",
                "logline",
                "scene_count_hint",
            }
            for key, value in brief.items():
                if key not in allowed_brief_keys:
                    continue
                if value in (None, "", []):
                    continue
                cleaned_brief[key] = value
            if cleaned_brief:
                sanitized["brief"] = cleaned_brief

        characters = plan.get("characters")
        if isinstance(characters, list):
            cleaned_chars: List[Dict[str, Any]] = []
            for raw in characters:
                if not isinstance(raw, dict):
                    continue
                item = {
                    "character_id": raw.get("character_id", ""),
                    "name": str(raw.get("name", "")).strip(),
                    "role": str(raw.get("role", "")).strip(),
                    "appearance": str(raw.get("appearance", "")).strip(),
                    "personality": str(raw.get("personality", "")).strip(),
                    "constraints": str(raw.get("constraints", "")).strip(),
                }
                voice = raw.get("voice")
                if isinstance(voice, dict):
                    item["voice"] = {
                        "voice_provider": str(voice.get("voice_provider", "")).strip(),
                        "voice_id": str(voice.get("voice_id", "")).strip(),
                        "speaking_style": str(voice.get("speaking_style", "")).strip(),
                    }
                if any(item.get(k) for k in ("name", "role", "appearance", "personality")):
                    cleaned_chars.append(item)
            if cleaned_chars:
                sanitized["characters"] = cleaned_chars

        world = plan.get("world")
        if isinstance(world, dict):
            cleaned_world: Dict[str, Any] = {}
            allowed_world_keys = {
                "setting",
                "background_style",
                "visual_references",
                "color_palette",
                "wardrobe_props",
                "camera_style",
                "lighting",
                "motion",
            }
            for key, value in world.items():
                if key not in allowed_world_keys:
                    continue
                if value in (None, "", []):
                    continue
                cleaned_world[key] = value
            if cleaned_world:
                sanitized["world"] = cleaned_world

        scene_plan_text = plan.get("scene_plan_text")
        scene_plan_items = []
        scene_plan = plan.get("scene_plan")
        if isinstance(scene_plan, list):
            lines = []
            for item in scene_plan:
                if isinstance(item, str) and item.strip():
                    scene_goal = item.strip()
                    scene_plan_items.append({"scene_goal": scene_goal, "scene_type": "auto"})
                    lines.append(scene_goal)
                elif isinstance(item, dict):
                    goal = str(
                        item.get("scene_goal")
                        or item.get("goal")
                        or item.get("description")
                        or ""
                    ).strip()
                    scene_type = str(item.get("scene_type") or "auto").strip()
                    if goal:
                        scene_plan_items.append(
                            {"scene_goal": goal, "scene_type": scene_type or "auto"}
                        )
                        lines.append(goal)
            if not isinstance(scene_plan_text, str):
                scene_plan_text = "\n".join(lines)
        elif isinstance(scene_plan, str) and not isinstance(scene_plan_text, str):
            scene_plan_text = scene_plan
        elif isinstance(plan.get("scenes"), list):
            lines = []
            for scene in plan["scenes"]:
                if not isinstance(scene, dict):
                    continue
                goal = str(
                    scene.get("scene_goal")
                    or scene.get("goal")
                    or scene.get("description")
                    or ""
                ).strip()
                scene_type = str(scene.get("scene_type") or "auto").strip()
                if goal:
                    scene_plan_items.append({"scene_goal": goal, "scene_type": scene_type})
                    lines.append(goal)
            if not isinstance(scene_plan_text, str):
                scene_plan_text = "\n".join(lines)
        if scene_plan_items:
            sanitized["scene_plan"] = scene_plan_items
        if isinstance(scene_plan_text, str) and scene_plan_text.strip():
            sanitized["scene_plan_text"] = scene_plan_text.strip()

        script_text = plan.get("script_text")
        if not isinstance(script_text, str):
            script = plan.get("script")
            if isinstance(script, str):
                script_text = script
            elif isinstance(script, list):
                lines = []
                for item in script:
                    if isinstance(item, str) and item.strip():
                        lines.append(item.strip())
                    elif isinstance(item, dict):
                        speaker = str(item.get("speaker") or item.get("speaker_id") or "").strip()
                        line = str(item.get("line") or "").strip()
                        if line:
                            lines.append(f"{speaker}: {line}" if speaker else line)
                script_text = "\n".join(lines)
        if isinstance(script_text, str) and script_text.strip():
            sanitized["script_text"] = script_text.strip()

        audio = plan.get("audio")
        if isinstance(audio, dict):
            cleaned_audio: Dict[str, Any] = {}
            if audio.get("music_mood") not in (None, ""):
                cleaned_audio["music_mood"] = str(audio.get("music_mood")).strip()
            sfx = audio.get("sound_effects")
            if isinstance(sfx, str):
                cleaned_audio["sound_effects"] = [
                    item.strip() for item in sfx.split(",") if item.strip()
                ]
            elif isinstance(sfx, list):
                cleaned_audio["sound_effects"] = [
                    str(item).strip() for item in sfx if str(item).strip()
                ]
            if cleaned_audio:
                sanitized["audio"] = cleaned_audio

        output = plan.get("output")
        if isinstance(output, dict):
            cleaned_output: Dict[str, Any] = {}
            allowed_output = {
                "resolution",
                "fps",
                "subtitles_enabled",
                "subtitle_style",
                "safe_areas",
                "file_format",
                "deliverables",
            }
            for key, value in output.items():
                if key not in allowed_output:
                    continue
                if value in (None, "", []):
                    continue
                cleaned_output[key] = value
            if cleaned_output:
                sanitized["output"] = cleaned_output

        assumptions = plan.get("assumptions")
        if isinstance(assumptions, list):
            cleaned_assumptions = []
            for item in assumptions:
                if isinstance(item, str) and item.strip():
                    cleaned_assumptions.append(item.strip())
                elif isinstance(item, dict):
                    reason = str(
                        item.get("reason") or item.get("note") or item.get("assumption") or ""
                    ).strip()
                    if reason:
                        cleaned_assumptions.append(reason)
            if cleaned_assumptions:
                sanitized["assumptions"] = cleaned_assumptions

        return sanitized
