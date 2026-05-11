from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types


class SceneBuilderService:
    SUPPORTED_SCENE_TYPES = {
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

    def __init__(
        self,
        api_key: str = "",
        model_name: str = "gemini-2.5-flash",
        timeout_seconds: int = 18,
        enabled: bool = True,
        client: Optional[Any] = None,
    ):
        key = str(api_key or "").strip()
        self.model_name = str(model_name or "gemini-2.5-flash").strip()
        self.timeout_seconds = int(timeout_seconds)
        self.enabled = bool(enabled) and bool(key) and key.lower() not in {
            "dummy_key",
            "test_key",
            "placeholder",
        }
        self.client = client
        if self.enabled and self.client is None:
            try:
                self.client = genai.Client(api_key=key)
            except Exception:
                self.enabled = False
                self.client = None

    def build_scenes(
        self,
        manifest: Dict[str, Any],
        rng: random.Random,
        *,
        replace: bool,
    ) -> List[Dict[str, Any]]:
        del rng  # Scenes are produced by Gemini model output.

        brief = manifest.get("brief", {})
        existing_scenes = manifest.get("scenes", [])
        existing_map = {
            item.get("scene_id"): item
            for item in existing_scenes
            if isinstance(item, dict) and item.get("scene_id")
        }
        scene_count = self._resolve_scene_count(
            brief,
            existing_scenes,
            replace=replace,
        )
        if not self.enabled or self.client is None:
            raise RuntimeError(
                "Scene generation requires Gemini. Set GEMINI_API_KEY and keep scene builder enabled."
            )
        model_plan, plan_error = self._plan_with_model(
            manifest=manifest,
            scene_count=scene_count,
            replace=replace,
        )
        if not model_plan:
            # When replacing or creating scenes from scratch, Gemini output is mandatory.
            if replace or not existing_map:
                raise RuntimeError(
                    "Gemini scene builder returned no usable scenes."
                    + (f" Details: {plan_error}" if plan_error else "")
                )
            # For non-replace syncs, preserve current scenes instead of hard-failing.
            model_plan = []
        total_scenes = scene_count if replace else max(scene_count, len(existing_map))
        missing_model_scenes: List[str] = []
        for index in range(total_scenes):
            scene_id = f"scene_{index + 1:03d}"
            existing = existing_map.get(scene_id)
            locked = bool(existing.get("locked")) if isinstance(existing, dict) else False
            preserve_existing = bool(existing) and (locked or not replace)
            has_model_scene = index < len(model_plan) and bool(model_plan[index])
            if not preserve_existing and not has_model_scene:
                missing_model_scenes.append(scene_id)
        if missing_model_scenes:
            raise RuntimeError(
                "Gemini scene builder returned fewer scenes than required: "
                + ", ".join(missing_model_scenes)
            )

        generated: List[Dict[str, Any]] = []
        previous_scene_id = ""

        for index in range(total_scenes):
            scene_id = f"scene_{index + 1:03d}"
            existing = existing_map.get(scene_id)
            model_scene = model_plan[index] if index < len(model_plan) else {}
            scene = self._compose_scene(
                manifest=manifest,
                index=index,
                scene_id=scene_id,
                model_scene=model_scene,
                existing=existing,
                previous_scene_id=previous_scene_id,
                replace=replace,
                model_used=bool(model_scene),
            )
            generated.append(scene)
            previous_scene_id = scene_id

        return generated

    def _compose_scene(
        self,
        *,
        manifest: Dict[str, Any],
        index: int,
        scene_id: str,
        model_scene: Dict[str, Any],
        existing: Optional[Dict[str, Any]],
        previous_scene_id: str,
        replace: bool,
        model_used: bool,
    ) -> Dict[str, Any]:
        brief = manifest.get("brief", {})
        world = manifest.get("world", {})
        existing_scene = existing if isinstance(existing, dict) else {}
        locked = bool(existing_scene.get("locked"))
        preserve_existing = bool(existing_scene) and (locked or not replace)
        primary, secondary = self._prioritize_sources(
            existing_scene,
            model_scene,
            preserve_existing,
        )
        cast_ids = self._cast_ids(manifest.get("characters", []))
        alias_to_id = self._character_alias_map(manifest.get("characters", []))

        scene_type = self._normalize_scene_type(
            self._text_with_priority(
                preferred=primary.get("scene_type"),
                fallback=secondary.get("scene_type"),
            ),
        )
        scene_goal = self._text_with_priority(
            preferred=primary.get("scene_goal")
            or primary.get("goal")
            or primary.get("scene_prompt")
            or primary.get("prompt")
            or primary.get("scene_description"),
            fallback=secondary.get("scene_goal")
            or secondary.get("goal")
            or secondary.get("scene_prompt")
            or secondary.get("prompt")
            or secondary.get("scene_description"),
            default="",
        )
        scene_description = self._text_with_priority(
            preferred=primary.get("scene_description")
            or primary.get("description")
            or scene_goal,
            fallback=secondary.get("scene_description")
            or secondary.get("description")
            or scene_goal,
            default=scene_goal,
        )
        scene_prompt = self._text_with_priority(
            preferred=primary.get("scene_prompt")
            or primary.get("prompt")
            or scene_goal
            or scene_description,
            fallback=secondary.get("scene_prompt")
            or secondary.get("prompt")
            or scene_goal
            or scene_description,
            default="",
        )
        if (
            scene_prompt
            and scene_goal
            and len(scene_prompt.split()) < 6
            and len(scene_goal.split()) > len(scene_prompt.split())
        ):
            scene_prompt = scene_goal
        if not scene_prompt and not preserve_existing:
            raise RuntimeError(
                f"Gemini scene builder returned an empty prompt for {scene_id}."
            )
        if not scene_goal:
            scene_goal = scene_prompt
        location_time = self._text_with_priority(
            preferred=primary.get("location_time"),
            fallback=secondary.get("location_time"),
            default=str(world.get("setting") or ""),
        )
        scene_cast = self._scene_cast(
            self._list_with_priority(
                preferred=primary.get("characters_in_scene"),
                fallback=secondary.get("characters_in_scene"),
            ),
            cast_ids,
            alias_to_id,
        )
        continuity = self._build_continuity(
            index=index,
            scene_id=scene_id,
            scene_type=scene_type,
            previous_scene_id=previous_scene_id,
            scene_cast=scene_cast,
            manifest=manifest,
            existing=existing_scene.get("continuity"),
        )

        conversation_lines = self._list_with_priority(
            preferred=primary.get("conversation"),
            fallback=secondary.get("conversation"),
            normalizer=self._normalize_conversation_lines,
            default=[],
        )
        dialogue = self._list_with_priority(
            preferred=primary.get("dialogue"),
            fallback=secondary.get("dialogue"),
            normalizer=self._normalize_dialogue,
            default=[],
        )
        if not dialogue and conversation_lines:
            dialogue = self._conversation_to_dialogue(
                conversation_lines,
                scene_cast,
                alias_to_id,
            )
        if not conversation_lines:
            conversation_lines = self._dialogue_to_conversation(dialogue)

        scene: Dict[str, Any] = {
            "scene_id": scene_id,
            "scene_prompt": scene_prompt,
            "conversation": conversation_lines,
            "scene_type": scene_type,
            "scene_goal": scene_goal,
            "scene_description": scene_description or scene_goal or scene_prompt,
            "spoken_direction": self._text_with_priority(
                preferred=primary.get("spoken_direction"),
                fallback=secondary.get("spoken_direction"),
                default="Natural, clear, conversational delivery.",
            ),
            "image_usage_instructions": self._text_with_priority(
                preferred=primary.get("image_usage_instructions"),
                fallback=secondary.get("image_usage_instructions"),
                default="Keep continuity with the previous scene and same character look.",
            ),
            "location_time": location_time,
            "characters_in_scene": scene_cast,
            "events": self._list_with_priority(
                preferred=primary.get("events"),
                fallback=secondary.get("events"),
                normalizer=self._normalize_events,
                default=[scene_prompt] if scene_prompt else [],
            ),
            "dialogue": dialogue,
            "narration": self._text_with_priority(
                preferred=primary.get("narration"),
                fallback=secondary.get("narration"),
            ),
            "shot_list": self._list_with_priority(
                preferred=primary.get("shot_list"),
                fallback=secondary.get("shot_list"),
                normalizer=self._normalize_shot_list,
                default=[],
            ),
            "reference_images": self._list_with_priority(
                preferred=primary.get("reference_images"),
                fallback=secondary.get("reference_images"),
                normalizer=self._normalize_text_list,
                default=[],
            ),
            "continuity": continuity,
            "prompt_memory": {
                "character_ids": scene_cast,
                "location_signature": location_time,
                "camera_style": str(world.get("camera_style") or ""),
                "tone": str(brief.get("tone") or ""),
            },
            "state": "locked" if locked else self._normalize_state(existing_scene.get("state")),
            "locked": locked,
            "builder": {
                "version": "scene_builder_simple_v1",
                "seed_fragment": "",
                "model_used": bool(model_used),
                "model_name": self.model_name if model_used else "",
            },
        }

        for key in ["active_variant_id", "state_history"]:
            if key in existing_scene:
                scene[key] = existing_scene.get(key)
        if locked:
            scene["state"] = "locked"
        return scene

    @staticmethod
    def _normalize_state(value: Any) -> str:
        state = str(value or "").strip().lower()
        allowed = {"planned", "generated", "approved", "locked", "stitched"}
        return state if state in allowed else "planned"

    @staticmethod
    def _prioritize_sources(
        existing_scene: Dict[str, Any],
        model_scene: Dict[str, Any],
        preserve_existing: bool,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if preserve_existing:
            return existing_scene, model_scene
        return model_scene, existing_scene

    def _resolve_scene_count(
        self,
        brief: Dict[str, Any],
        existing_scenes: List[Any],
        *,
        replace: bool,
    ) -> int:
        duration = self._safe_int(brief.get("duration_seconds"), default=15)
        scene_count = max(1, round(duration / 8))
        scene_count_hint = self._safe_int(brief.get("scene_count_hint"), default=0)
        existing_count = sum(
            1
            for item in existing_scenes
            if isinstance(item, dict) and str(item.get("scene_id", "")).strip()
        )
        if scene_count_hint > 0:
            return max(1, scene_count_hint) if replace else max(scene_count_hint, existing_count)
        return scene_count if replace else max(scene_count, existing_count)

    def _build_continuity(
        self,
        *,
        index: int,
        scene_id: str,
        scene_type: str,
        previous_scene_id: str,
        scene_cast: List[str],
        manifest: Dict[str, Any],
        existing: Any = None,
    ) -> Dict[str, Any]:
        continuity = existing if isinstance(existing, dict) else {}
        must_follow_previous = bool(previous_scene_id)
        opening_anchor = (
            "previous_scene_end_frame" if must_follow_previous else "character_reference_images"
        )
        merged = {
            "scene_id": scene_id,
            "chain_index": index + 1,
            "from_scene_id": previous_scene_id or None,
            "must_follow_previous_frame": must_follow_previous,
            "opening_anchor": opening_anchor,
            "strategy": "last_frame_plus_next_goal",
            "character_reference_ids": list(scene_cast),
            "style_lock": {
                "tone": str(manifest.get("brief", {}).get("tone") or ""),
                "camera_style": str(manifest.get("world", {}).get("camera_style") or ""),
                "lighting": str(manifest.get("world", {}).get("lighting") or ""),
            },
            "scene_type": scene_type,
        }
        if continuity:
            merged.update(continuity)
        merged["scene_id"] = scene_id
        merged["chain_index"] = index + 1
        merged["from_scene_id"] = previous_scene_id or None
        merged["must_follow_previous_frame"] = must_follow_previous
        merged["opening_anchor"] = opening_anchor
        merged["character_reference_ids"] = list(scene_cast)
        merged["scene_type"] = scene_type
        return merged

    def _plan_with_model(
        self,
        *,
        manifest: Dict[str, Any],
        scene_count: int,
        replace: bool,
    ) -> Tuple[List[Dict[str, Any]], str]:
        prompt = self._build_model_prompt(
            manifest=manifest,
            scene_count=scene_count,
            replace=replace,
        )
        last_error = ""
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            request_prompt = prompt
            if attempt > 1 and last_error:
                request_prompt = (
                    f"{prompt}\n\nPrevious response failed requirements:\n"
                    f"- {last_error}\n"
                    "Rewrite the full JSON and fix every issue."
                )
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=request_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type="application/json",
                    ),
                )
            except Exception as error:
                last_error = f"Gemini request failed on attempt {attempt}/{max_attempts}: {error}"
                continue

            raw_text = self._extract_text(response)
            if not raw_text:
                last_error = (
                    f"Gemini returned empty text on attempt {attempt}/{max_attempts}."
                )
                continue

            parsed = self._parse_json_like(raw_text)
            if not isinstance(parsed, dict):
                preview = raw_text.strip().replace("\n", " ")[:220]
                last_error = (
                    f"Gemini returned non-JSON content on attempt {attempt}/{max_attempts}: {preview}"
                )
                continue

            scenes = self._extract_scene_candidates(parsed)
            if not scenes:
                preview = raw_text.strip().replace("\n", " ")[:220]
                keys = ", ".join(list(parsed.keys())[:10])
                last_error = (
                    "Gemini JSON contained no recognizable scene list "
                    f"on attempt {attempt}/{max_attempts} "
                    f"(keys: {keys or 'none'}): {preview}"
                )
                continue

            sanitized: List[Dict[str, Any]] = []
            for index, item in enumerate(scenes):
                if isinstance(item, str):
                    prompt_text = item.strip()
                    if not prompt_text:
                        continue
                    scene_id = f"scene_{index + 1:03d}"
                    conversation = self._normalize_conversation_lines([])
                    dialogue = self._normalize_dialogue([])
                    sanitized.append(
                        {
                            "scene_id": scene_id,
                            "scene_type": "auto",
                            "scene_prompt": prompt_text,
                            "scene_goal": prompt_text,
                            "scene_description": prompt_text,
                            "spoken_direction": "",
                            "image_usage_instructions": "",
                            "location_time": "",
                            "characters_in_scene": [],
                            "events": [prompt_text],
                            "conversation": conversation,
                            "dialogue": dialogue,
                            "shot_list": [],
                            "reference_images": [],
                            "narration": "",
                        }
                    )
                    continue
                if not isinstance(item, dict):
                    continue
                scene_id = str(item.get("scene_id") or f"scene_{index + 1:03d}").strip()
                if not scene_id:
                    continue
                scene_goal = str(
                    item.get("scene_goal")
                    or item.get("goal")
                    or item.get("scene_prompt")
                    or item.get("prompt")
                    or item.get("scene_description")
                    or item.get("description")
                    or ""
                ).strip()
                scene_description = str(
                    item.get("scene_description")
                    or item.get("description")
                    or scene_goal
                    or ""
                ).strip()
                scene_prompt = str(
                    item.get("scene_prompt")
                    or item.get("prompt")
                    or scene_goal
                    or scene_description
                    or ""
                ).strip()
                conversation = self._normalize_conversation_lines(
                    item.get("conversation") or item.get("conversation_lines")
                )
                dialogue = self._normalize_dialogue(item.get("dialogue"))
                if not dialogue and conversation:
                    dialogue = self._conversation_to_dialogue(conversation, [], {})
                if not conversation and dialogue:
                    conversation = self._dialogue_to_conversation(dialogue)
                events = self._normalize_events(item.get("events"))
                if not scene_prompt and events:
                    scene_prompt = events[0]
                if not scene_goal:
                    scene_goal = scene_prompt
                if not scene_description:
                    scene_description = scene_goal
                if not scene_prompt:
                    scene_prompt = str(item.get("narration") or "").strip()
                if not scene_prompt:
                    continue
                if (
                    scene_prompt
                    and scene_goal
                    and len(scene_prompt.split()) < 6
                    and len(scene_goal.split()) > len(scene_prompt.split())
                ):
                    scene_prompt = scene_goal
                sanitized.append(
                    {
                        "scene_id": scene_id,
                        "scene_type": self._normalize_scene_type(item.get("scene_type")),
                        "scene_prompt": scene_prompt,
                        "scene_goal": scene_goal or scene_prompt,
                        "scene_description": scene_description or scene_goal or scene_prompt,
                        "spoken_direction": str(
                            item.get("spoken_direction") or item.get("what_should_be_said") or ""
                        ).strip(),
                        "image_usage_instructions": str(
                            item.get("image_usage_instructions") or item.get("image_usage") or ""
                        ).strip(),
                        "location_time": str(item.get("location_time") or "").strip(),
                        "characters_in_scene": self._normalize_text_list(
                            item.get("characters_in_scene")
                            or item.get("characters")
                            or item.get("cast")
                        ),
                        "events": events or [scene_goal or scene_prompt],
                        "conversation": conversation,
                        "dialogue": dialogue,
                        "shot_list": self._normalize_shot_list(item.get("shot_list")),
                        "reference_images": self._normalize_text_list(
                            item.get("reference_images")
                        ),
                        "narration": str(item.get("narration") or "").strip(),
                    }
                )
            if sanitized:
                fit_error = self._validate_scene_plan(
                    manifest=manifest,
                    scenes=sanitized,
                    scene_count=scene_count,
                )
                if fit_error:
                    last_error = fit_error
                    continue
                return sanitized, ""

            last_error = (
                f"Gemini JSON contained no valid scene objects on attempt {attempt}/{max_attempts}."
            )
        return [], last_error

    def _validate_scene_plan(
        self,
        *,
        manifest: Dict[str, Any],
        scenes: List[Dict[str, Any]],
        scene_count: int,
    ) -> str:
        required_count = max(1, scene_count)
        if len(scenes) < required_count:
            return f"Expected at least {required_count} scenes, got {len(scenes)}."

        trimmed = scenes[:required_count]
        combined = " ".join(
            " ".join(
                [
                    str(scene.get("scene_prompt") or ""),
                    str(scene.get("scene_goal") or ""),
                    str(scene.get("scene_description") or ""),
                    " ".join(scene.get("events", []) or []),
                    " ".join(scene.get("conversation", []) or []),
                ]
            )
            for scene in trimmed
        ).lower()

        description = str(manifest.get("brief", {}).get("description") or "").lower()
        if "splendoure" in description and "splendoure" not in combined:
            return "Missing brand mention 'Splendoure' in generated scenes."

        return ""

    @staticmethod
    def _extract_scene_candidates(parsed: Dict[str, Any]) -> List[Any]:
        preferred_keys = (
            "scenes",
            "scene_list",
            "scene_plan",
            "scene_outline",
            "scene_breakdown",
            "items",
        )
        for key in preferred_keys:
            value = parsed.get(key)
            if isinstance(value, list):
                return value

        for key in ("data", "result", "output", "response"):
            nested = parsed.get(key)
            if isinstance(nested, list):
                return nested
            if isinstance(nested, dict):
                nested_scenes = SceneBuilderService._extract_scene_candidates(nested)
                if nested_scenes:
                    return nested_scenes

        keyed_scenes: List[Dict[str, Any]] = []
        for key, value in parsed.items():
            normalized_key = str(key or "").strip().lower()
            if not normalized_key.startswith("scene_"):
                continue
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("scene_id", str(key))
                keyed_scenes.append(item)
            elif isinstance(value, str):
                prompt = value.strip()
                if prompt:
                    keyed_scenes.append(
                        {
                            "scene_id": str(key),
                            "scene_prompt": prompt,
                        }
                    )
        if keyed_scenes:
            keyed_scenes.sort(key=lambda item: str(item.get("scene_id", "")))
            return keyed_scenes

        for value in parsed.values():
            if isinstance(value, list) and any(
                isinstance(item, (dict, str)) for item in value
            ):
                return value
        return []

    def _build_model_prompt(
        self,
        *,
        manifest: Dict[str, Any],
        scene_count: int,
        replace: bool,
    ) -> str:
        brief = manifest.get("brief", {})
        world = manifest.get("world", {})
        characters = manifest.get("characters", [])
        existing_scenes = manifest.get("scenes", [])
        context = {
            "project_seed": manifest.get("project_seed", ""),
            "replace_mode": bool(replace),
            "requested_scene_count": scene_count,
            "description": str(brief.get("description") or ""),
            "brief": {
                "video_type": brief.get("video_type", ""),
                "goal": brief.get("goal", ""),
                "target_audience": brief.get("target_audience", ""),
                "platform": brief.get("platform", ""),
                "duration_seconds": brief.get("duration_seconds", 15),
                "aspect_ratio": brief.get("aspect_ratio", "9:16"),
                "tone": brief.get("tone", ""),
                "language": brief.get("language", ""),
                "accent": brief.get("accent", ""),
                "logline": brief.get("logline", ""),
            },
            "world": {
                "setting": world.get("setting", ""),
                "background_style": world.get("background_style", ""),
                "camera_style": world.get("camera_style", ""),
                "lighting": world.get("lighting", ""),
                "motion": world.get("motion", ""),
            },
            "characters": [
                {
                    "character_id": item.get("character_id", ""),
                    "name": item.get("name", ""),
                    "role": item.get("role", ""),
                    "appearance": item.get("appearance", ""),
                    "personality": item.get("personality", ""),
                    "reference_images": item.get("reference_images", []),
                }
                for item in characters
                if isinstance(item, dict)
            ],
            "current_scenes": [
                {
                    "scene_id": item.get("scene_id", ""),
                    "scene_goal": item.get("scene_goal", ""),
                    "scene_type": item.get("scene_type", ""),
                    "locked": bool(item.get("locked", False)),
                }
                for item in existing_scenes
                if isinstance(item, dict)
            ],
        }
        return (
            "You are the scene builder for a short-form video tool.\n"
            "Use the user's description to generate a coherent ad arc that fits the requested number of scenes.\n"
            "Return strict JSON only with this shape:\n"
            "{\n"
            '  "scenes": [\n'
            "    {\n"
            '      "scene_id": "scene_001",\n'
            '      "scene_type": "hook|problem|reframe|solution|payoff|cta|b_roll|testimonial|tutorial|transition|custom",\n'
            '      "scene_prompt": "1-3 sentence visual scene prompt",\n'
            '      "scene_goal": "what this scene must achieve for the ad argument",\n'
            '      "scene_description": "specific visual and emotional detail",\n'
            '      "events": ["beat 1", "beat 2", "beat 3"],\n'
            '      "conversation": ["Speaker: line"],\n'
            '      "dialogue": [{"speaker_id":"char_id","line":"text"}],\n'
            '      "location_time": "specific setting and time of day",\n'
            '      "characters_in_scene": ["character_id or character_name"]\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Rules:\n"
            f"1) Return exactly {scene_count} scenes.\n"
            "2) Scene 1 must be a strong hook.\n"
            "3) Cover the full ad angle from problem to resolution inside the requested scene count.\n"
            "4) Every scene after scene_001 must progress naturally from previous scene intent.\n"
            "5) Respect provided characters and avoid inventing new named characters.\n"
            "6) scene_prompt and scene_goal must be specific, not generic one-liners.\n"
            "7) If description contains a brand/product name, include it in solution/cta scenes.\n"
            "8) Provide 1-3 short conversation lines only when useful.\n"
            f"9) Compress and prioritize the strongest beats to fit exactly {scene_count} scenes.\n\n"
            f"Context JSON:\n{json.dumps(context)}\n"
        )

    def _scene_cast(
        self,
        raw_scene_cast: Any,
        cast_ids: List[str],
        alias_to_id: Dict[str, str],
    ) -> List[str]:
        if not cast_ids:
            return []
        requested: List[str] = []
        if isinstance(raw_scene_cast, list):
            for item in raw_scene_cast:
                raw = str(item or "").strip()
                if not raw:
                    continue
                key = raw.lower().replace(" ", "_")
                mapped = alias_to_id.get(key) or (raw if raw in cast_ids else "")
                if mapped and mapped not in requested:
                    requested.append(mapped)
        if requested:
            return requested[:3]
        return cast_ids[:3]

    @staticmethod
    def _cast_ids(characters: List[Any]) -> List[str]:
        ids: List[str] = []
        for item in characters:
            if not isinstance(item, dict):
                continue
            character_id = str(item.get("character_id", "")).strip()
            if character_id and character_id not in ids:
                ids.append(character_id)
        return ids

    @staticmethod
    def _character_alias_map(characters: List[Any]) -> Dict[str, str]:
        aliases: Dict[str, str] = {}
        for item in characters:
            if not isinstance(item, dict):
                continue
            character_id = str(item.get("character_id", "")).strip()
            if not character_id:
                continue
            aliases[character_id.lower()] = character_id
            aliases[character_id.lower().replace(" ", "_")] = character_id
            aliases[character_id.lower().replace("char_", "")] = character_id
            name = str(item.get("name", "")).strip().lower()
            if name:
                aliases[name] = character_id
                aliases[name.replace(" ", "_")] = character_id
        return aliases

    def _normalize_scene_type(self, value: Any) -> str:
        text = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
        if text in self.SUPPORTED_SCENE_TYPES:
            return text
        return "auto"

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _text_with_priority(preferred: Any, fallback: Any, default: str = "") -> str:
        preferred_text = str(preferred or "").strip()
        if preferred_text:
            return preferred_text
        fallback_text = str(fallback or "").strip()
        if fallback_text:
            return fallback_text
        return default

    def _list_with_priority(
        self,
        *,
        preferred: Any,
        fallback: Any,
        normalizer=None,
        default: Optional[List[Any]] = None,
    ) -> List[Any]:
        normalize = normalizer or self._normalize_text_list
        preferred_list = normalize(preferred)
        if preferred_list:
            return preferred_list
        fallback_list = normalize(fallback)
        if fallback_list:
            return fallback_list
        return list(default or [])

    @staticmethod
    def _normalize_text_list(raw: Any) -> List[str]:
        if not isinstance(raw, list):
            return []
        values: List[str] = []
        for item in raw:
            text = str(item or "").strip()
            if text:
                values.append(text)
        return values

    def _normalize_events(self, raw: Any) -> List[str]:
        return self._normalize_text_list(raw)[:4]

    def _normalize_conversation_lines(self, raw: Any) -> List[str]:
        if not isinstance(raw, list):
            return []
        lines: List[str] = []
        for item in raw:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    lines.append(text)
                continue
            if isinstance(item, dict):
                speaker = str(item.get("speaker") or item.get("speaker_id") or "").strip()
                line = str(item.get("line") or "").strip()
                if not line:
                    continue
                if speaker:
                    lines.append(f"{speaker}: {line}")
                else:
                    lines.append(line)
        return lines[:6]

    def _conversation_to_dialogue(
        self,
        conversation_lines: List[str],
        scene_cast: List[str],
        alias_to_id: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        if not conversation_lines:
            return []
        speakers = scene_cast[:] if scene_cast else ["narrator"]
        dialogue: List[Dict[str, Any]] = []
        cursor = 0.0
        for index, raw_line in enumerate(conversation_lines):
            text = str(raw_line or "").strip()
            if not text:
                continue
            speaker = speakers[index % len(speakers)]
            line = text
            if ":" in text:
                maybe_speaker, maybe_line = text.split(":", 1)
                maybe_line = maybe_line.strip()
                if maybe_line:
                    line = maybe_line
                    key = maybe_speaker.strip().lower().replace(" ", "_")
                    mapped = alias_to_id.get(key)
                    if mapped:
                        speaker = mapped
                    elif key in {"narrator", "voiceover", "on-screen_text"}:
                        speaker = key
            word_count = max(1, len(line.split()))
            duration = max(1.2, min(4.5, word_count * 0.34))
            dialogue.append(
                {
                    "speaker_id": speaker,
                    "line": line,
                    "emotion": "natural",
                    "pause_notes": "",
                    "start_second": round(cursor, 2),
                    "end_second": round(cursor + duration, 2),
                }
            )
            cursor += duration
        return dialogue

    @staticmethod
    def _dialogue_to_conversation(dialogue: List[Dict[str, Any]]) -> List[str]:
        lines: List[str] = []
        for item in dialogue or []:
            if not isinstance(item, dict):
                continue
            line = str(item.get("line") or "").strip()
            if not line:
                continue
            speaker = str(item.get("speaker_id") or "narrator").strip()
            lines.append(f"{speaker}: {line}")
        return lines[:6]

    def _normalize_dialogue(self, raw: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        lines: List[Dict[str, Any]] = []
        local_cursor = 0.0
        for item in raw:
            if not isinstance(item, dict):
                continue
            text = str(item.get("line") or "").strip()
            if not text:
                continue
            speaker = str(item.get("speaker_id") or item.get("speaker") or "narrator").strip()
            emotion = str(item.get("emotion") or "natural").strip()
            pause_notes = str(item.get("pause_notes") or "").strip()
            duration = 2.8
            try:
                maybe_duration = float(item.get("duration_seconds", 0))
                if maybe_duration > 0:
                    duration = max(1.2, min(4.5, maybe_duration))
                else:
                    start_in = float(item.get("start_second", 0))
                    end_in = float(item.get("end_second", 0))
                    if end_in > start_in:
                        duration = max(1.2, min(4.5, end_in - start_in))
            except Exception:
                pass
            lines.append(
                {
                    "speaker_id": speaker or "narrator",
                    "line": text,
                    "emotion": emotion,
                    "pause_notes": pause_notes,
                    "start_second": round(local_cursor, 2),
                    "end_second": round(local_cursor + duration, 2),
                }
            )
            local_cursor += duration
        return lines

    def _normalize_shot_list(self, raw: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        shots: List[Dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            duration = 2.8
            try:
                duration = max(1.0, min(8.0, float(item.get("duration_seconds", 2.8))))
            except Exception:
                duration = 2.8
            shots.append(
                {
                    "shot_type": str(item.get("shot_type") or "medium").strip(),
                    "camera_move": str(item.get("camera_move") or "subtle pan").strip(),
                    "subject_focus": str(item.get("subject_focus") or "primary action").strip(),
                    "duration_seconds": round(duration, 2),
                    "on_screen_text": str(item.get("on_screen_text") or "").strip(),
                    "music_mood": str(item.get("music_mood") or "uplifting").strip(),
                    "sound_effects": self._normalize_text_list(item.get("sound_effects")),
                    "transition_out": str(item.get("transition_out") or "cut").strip(),
                }
            )
        return shots

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
            content = content[:-3].strip()

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
