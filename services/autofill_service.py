from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple

from schemas import utc_now_iso
from services.scene_builder_service import SceneBuilderService


class ManifestAutoFillService:
    STEP_SALTS = {
        "brief": "brief",
        "characters": "characters",
        "world": "world",
        "scenes": "scenes",
        "script": "dialogue",
        "audio": "audio",
        "output": "captions",
        "review": "review",
    }

    def __init__(
        self,
        scene_builder_service: Optional[SceneBuilderService] = None,
    ):
        self.scene_builder_service = scene_builder_service or SceneBuilderService()

    def autofill_step(
        self,
        manifest: Dict[str, Any],
        step_name: str,
        reason: str = "step_skip",
        *,
        force_scene_rebuild: bool = False,
    ) -> Dict[str, Any]:
        manifest.setdefault("surprise_counters", {})
        rng, seed_fragment = self._rng_for_step(manifest, step_name)

        if step_name == "brief":
            self._fill_brief(manifest, rng, seed_fragment, reason)
        elif step_name == "characters":
            self._fill_characters(manifest, rng, seed_fragment, reason, replace=False)
        elif step_name == "world":
            self._fill_world(manifest, rng, seed_fragment, reason)
        elif step_name == "scenes":
            self._fill_scenes(
                manifest,
                rng,
                seed_fragment,
                reason,
                replace=force_scene_rebuild,
            )
        elif step_name == "script":
            self._fill_script(manifest, rng, seed_fragment, reason, replace=False)
        elif step_name == "audio":
            self._fill_audio(manifest, rng, seed_fragment, reason)
        elif step_name == "output":
            self._fill_output(manifest, rng, seed_fragment, reason)
        elif step_name == "review":
            self.autofill_all(
                manifest,
                reason=reason,
                force_scene_rebuild=force_scene_rebuild,
            )
        else:
            raise ValueError(f"Unknown step '{step_name}'")

        manifest["updated_at"] = utc_now_iso()
        return manifest

    def surprise_step(self, manifest: Dict[str, Any], step_name: str) -> Dict[str, Any]:
        counters = manifest.setdefault("surprise_counters", {})
        counters[step_name] = int(counters.get(step_name, 0)) + 1

        rng, seed_fragment = self._rng_for_step(
            manifest, step_name, surprise_index=counters[step_name]
        )
        reason = "surprise_me"

        if step_name == "brief":
            self._fill_brief(manifest, rng, seed_fragment, reason, replace_missing=False)
        elif step_name == "characters":
            self._fill_characters(manifest, rng, seed_fragment, reason, replace=True)
        elif step_name == "world":
            self._fill_world(manifest, rng, seed_fragment, reason, force=True)
        elif step_name == "scenes":
            self._fill_scenes(manifest, rng, seed_fragment, reason, replace=True)
        elif step_name == "script":
            self._fill_script(manifest, rng, seed_fragment, reason, replace=True)
        elif step_name == "audio":
            self._fill_audio(manifest, rng, seed_fragment, reason, force=True)
        elif step_name == "output":
            self._fill_output(manifest, rng, seed_fragment, reason, force=True)
        elif step_name == "review":
            self.autofill_all(manifest, reason=reason)
        else:
            raise ValueError(f"Unknown step '{step_name}'")

        manifest["updated_at"] = utc_now_iso()
        return manifest

    def autofill_all(
        self,
        manifest: Dict[str, Any],
        reason: str = "review_autofill",
        *,
        force_scene_rebuild: bool = False,
    ) -> Dict[str, Any]:
        ordered_steps = [
            "brief",
            "characters",
            "world",
            "scenes",
            "script",
            "audio",
            "output",
        ]
        for step in ordered_steps:
            self.autofill_step(
                manifest,
                step,
                reason=reason,
                force_scene_rebuild=force_scene_rebuild,
            )
        self.sync_character_bible(manifest, reason=reason)
        self._sync_scene_characters_with_cast(manifest)
        return manifest

    def _rng_for_step(
        self, manifest: Dict[str, Any], step_name: str, surprise_index: int = 0
    ) -> Tuple[random.Random, str]:
        project_seed = str(manifest.get("project_seed", "default-seed"))
        salt = self.STEP_SALTS.get(step_name, step_name)
        key = f"{project_seed}:{salt}:{surprise_index}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return random.Random(int(digest[:16], 16)), digest[:12]

    def _fill_brief(
        self,
        manifest: Dict[str, Any],
        rng: random.Random,
        seed_fragment: str,
        reason: str,
        replace_missing: bool = True,
    ) -> None:
        brief = manifest.setdefault("brief", {})

        defaults = {
            "video_type": "social skit",
            "goal": "Make viewers curious and ready to act.",
            "target_audience": "young professionals",
            "platform": "TikTok",
            "duration_seconds": 15,
            "aspect_ratio": "9:16",
            "tone": "engaging",
            "language": "English",
            "accent": "neutral American",
            "brand_rules": "Avoid unsafe claims and impersonation.",
        }
        for field, value in defaults.items():
            if replace_missing and brief.get(field) in (None, "", []):
                brief[field] = value
                self._record(manifest, f"brief.{field}", value, reason, seed_fragment)

        if not brief.get("logline"):
            logline = self._build_logline(brief, rng)
            brief["logline"] = logline
            self._record(manifest, "brief.logline", logline, reason, seed_fragment)

    def _fill_characters(
        self,
        manifest: Dict[str, Any],
        rng: random.Random,
        seed_fragment: str,
        reason: str,
        replace: bool,
    ) -> None:
        characters = manifest.setdefault("characters", [])
        if characters and not replace:
            for character in characters:
                self._ensure_character_voice(manifest, character, seed_fragment, reason)
            self.sync_character_bible(manifest, reason=reason)
            self._sync_scene_characters_with_cast(manifest)
            return

        brief = manifest.get("brief", {})
        video_type = str(brief.get("video_type", "social skit")).lower()
        if "tutorial" in video_type or "explainer" in video_type:
            roles = ["host", "customer"]
        elif "testimonial" in video_type:
            roles = ["customer", "narrator"]
        else:
            roles = ["protagonist", "friend"]

        personalities = ["confident", "curious", "witty", "calm", "energetic"]
        attire = ["smart casual", "streetwear", "business casual", "minimalist fashion"]

        generated: List[Dict[str, Any]] = []
        for idx, role in enumerate(roles, start=1):
            char_id = f"char_{idx}"
            character = {
                "character_id": char_id,
                "name": f"Character {idx}",
                "role": role,
                "appearance": f"Age 22-35, {rng.choice(attire)}, distinctive smile.",
                "personality": rng.choice(personalities),
                "constraints": "Avoid resemblance to real people.",
                "voice": {},
            }
            self._ensure_character_voice(manifest, character, seed_fragment, reason)
            generated.append(character)

        manifest["characters"] = generated
        self._record(
            manifest,
            "characters",
            generated,
            reason,
            seed_fragment,
        )
        self.sync_character_bible(manifest, reason=reason)
        self._sync_scene_characters_with_cast(manifest)

    def _ensure_character_voice(
        self,
        manifest: Dict[str, Any],
        character: Dict[str, Any],
        seed_fragment: str,
        reason: str,
    ) -> None:
        voice = character.setdefault("voice", {})
        if not voice.get("voice_provider"):
            voice["voice_provider"] = "native"
        if not voice.get("voice_id"):
            voice["voice_id"] = f"native_{character.get('character_id', 'voice')}"
        if not voice.get("speaking_style"):
            personality = character.get("personality", "clear")
            voice["speaking_style"] = f"{personality}, medium pace, expressive"

        self._record(
            manifest,
            f"characters.{character.get('character_id', 'unknown')}.voice",
            voice,
            reason,
            seed_fragment,
        )

    def _fill_world(
        self,
        manifest: Dict[str, Any],
        rng: random.Random,
        seed_fragment: str,
        reason: str,
        force: bool = False,
    ) -> None:
        brief = manifest.get("brief", {})
        world = manifest.setdefault("world", {})
        platform = str(brief.get("platform", "tiktok")).lower()
        tone = str(brief.get("tone", "engaging")).lower()

        defaults = {
            "setting": "Modern city street, golden hour",
            "background_style": "cinematic realistic",
            "visual_references": [],
            "color_palette": "warm amber, deep blue, high contrast",
            "wardrobe_props": "Everyday fashionable outfits, smartphone, shopping bag",
            "camera_style": "dynamic handheld with controlled movement",
            "lighting": "soft daylight with cinematic highlights",
            "motion": "energetic with fast hooks",
        }

        if "linkedin" in platform:
            defaults["camera_style"] = "tripod and steady medium shots"
            defaults["motion"] = "calm, deliberate pans"
        elif "youtube" in platform:
            defaults["camera_style"] = "mixed wide and close-up coverage"
            defaults["motion"] = "balanced cuts and smooth pans"

        if "luxury" in tone:
            defaults["color_palette"] = "gold, black, ivory"
            defaults["lighting"] = "dramatic soft key light"
        elif "gritty" in tone:
            defaults["lighting"] = "high contrast, moody shadows"
            defaults["background_style"] = "gritty cinematic realism"

        for field, value in defaults.items():
            if force or world.get(field) in (None, "", []):
                world[field] = value
                self._record(manifest, f"world.{field}", value, reason, seed_fragment)

        if force and not world.get("visual_references"):
            world["visual_references"] = [
                "auto://moodboard/cinematic-street-style",
                "auto://moodboard/platform-safe-layout",
            ]
        self.sync_character_bible(manifest, reason=reason)

    def _fill_scenes(
        self,
        manifest: Dict[str, Any],
        rng: random.Random,
        seed_fragment: str,
        reason: str,
        replace: bool,
    ) -> None:
        manifest["scenes"] = self.scene_builder_service.build_scenes(
            manifest=manifest,
            rng=rng,
            replace=replace,
        )
        self._sync_scene_characters_with_cast(manifest)
        self._record(manifest, "scenes", manifest["scenes"], reason, seed_fragment)

    def _fill_script(
        self,
        manifest: Dict[str, Any],
        rng: random.Random,
        seed_fragment: str,
        reason: str,
        replace: bool,
    ) -> None:
        scenes = manifest.setdefault("scenes", [])
        characters = manifest.get("characters", [])
        if not characters:
            self._fill_characters(manifest, rng, seed_fragment, reason, replace=True)
            characters = manifest.get("characters", [])

        self._sync_scene_characters_with_cast(manifest)
        cast_ids = [
            item.get("character_id")
            for item in characters
            if isinstance(item, dict) and item.get("character_id")
        ]
        first_character_id = cast_ids[0] if cast_ids else "char_1"
        second_character_id = cast_ids[1] if len(cast_ids) > 1 else first_character_id
        audience = str(manifest.get("brief", {}).get("target_audience", "viewers")).strip()
        goal = str(manifest.get("brief", {}).get("goal", "take action")).strip()
        platform = str(manifest.get("brief", {}).get("platform", "social")).strip()

        timeline_entries: List[Dict[str, Any]] = []
        total_time = 0.0

        for index, scene in enumerate(scenes):
            if scene.get("dialogue"):
                self._retime_scene_dialogue(scene, total_time)

            if scene.get("locked") and scene.get("dialogue") and not replace:
                total_time += self._scene_duration(scene)
                continue

            scene_cast = [
                item
                for item in scene.get("characters_in_scene", [])
                if item in cast_ids
            ]
            if not scene_cast:
                scene_cast = [first_character_id]
                if second_character_id != first_character_id:
                    scene_cast.append(second_character_id)
                scene["characters_in_scene"] = scene_cast
            speaker_a = scene_cast[0] if scene_cast else first_character_id
            speaker_b = scene_cast[1] if len(scene_cast) > 1 else speaker_a
            scene_type = str(scene.get("scene_type", "auto")).lower()
            scene_goal = str(scene.get("scene_goal", "")).strip()
            short_goal = self._short_scene_goal(scene_goal)

            if replace or not scene.get("dialogue"):
                lead_templates = {
                    "hook": [
                        f"New city, introvert, and zero plans? This is for you.",
                        f"If making friends feels awkward, you are not alone.",
                        f"Pause for two seconds, this could change your week.",
                    ],
                    "problem": [
                        f"You want to go out, but starting conversations drains you.",
                        f"Most {audience} keep scrolling because social plans feel hard to enter.",
                        f"The pain is not going out, it is figuring out where to start.",
                    ],
                    "reframe": [
                        f"The shift is simple: join plans first, talk later.",
                        f"Instead of cold intros, step into existing plans.",
                        f"Small talk is optional when the plan already exists.",
                    ],
                    "solution": [
                        f"Open the app, swipe real plans, and pick one that fits tonight.",
                        f"You browse actual dinners, meetups, and city plans in seconds.",
                        f"Find the plan, tap join, and show up without overthinking.",
                    ],
                    "payoff": [
                        f"No pressure, no awkward start, just people doing things together.",
                        f"That is how social life becomes easier and more natural.",
                        f"Now going out feels simple, not stressful.",
                    ],
                    "cta": [
                        f"It is called Splendoure. Join your next social plan now.",
                        f"Download now and move from scrolling to showing up.",
                        f"Try it today and join a plan on {platform}.",
                    ],
                    "auto": [
                        f"{short_goal}",
                        f"{goal}",
                        f"Here is the next step that moves this story forward.",
                    ],
                }
                follow_templates = {
                    "hook": [
                        "Stay for the next part and see how this works.",
                        "You will see the exact flow in the next seconds.",
                        "Let me show you the easiest path right now.",
                    ],
                    "problem": [
                        "If this sounds familiar, the fix is in the next scene.",
                        "This is exactly why most people feel stuck socially.",
                        "You do not need to force conversations first.",
                    ],
                    "reframe": [
                        "That one mindset shift changes everything.",
                        "Now you are optimizing for plans, not pressure.",
                        "This keeps it social without making it awkward.",
                    ],
                    "solution": [
                        "Swipe, join, and walk in with context already set.",
                        "Each plan tells you what to expect before you go.",
                        "That is the easiest way to be social in a new city.",
                    ],
                    "payoff": [
                        "You feel included before you even arrive.",
                        "The outcome is confidence without forced small talk.",
                        "This is why people keep using it week after week.",
                    ],
                    "cta": [
                        "Download and join your first plan today.",
                        "Tap in now and start showing up socially.",
                        "If this is you, try Splendoure now.",
                    ],
                    "auto": [
                        "This scene builds momentum for the final action.",
                        "Keep the flow and move to the next beat.",
                        "This connects directly to the core promise.",
                    ],
                }
                lead = rng.choice(lead_templates.get(scene_type, lead_templates["auto"]))
                follow = rng.choice(follow_templates.get(scene_type, follow_templates["auto"]))
                scene["dialogue"] = [
                    {
                        "speaker_id": speaker_a,
                        "line": lead,
                        "emotion": "confident",
                        "pause_notes": "short beat after hook",
                        "start_second": round(total_time, 2),
                        "end_second": round(total_time + 2.8, 2),
                    },
                    {
                        "speaker_id": speaker_b,
                        "line": follow,
                        "emotion": "encouraging",
                        "pause_notes": "finish with CTA emphasis",
                        "start_second": round(total_time + 2.8, 2),
                        "end_second": round(total_time + 5.8, 2),
                    },
                ]
                self._record(
                    manifest,
                    f"scenes.{scene.get('scene_id')}.dialogue",
                    scene["dialogue"],
                    reason,
                    seed_fragment,
                )

            if replace or not scene.get("shot_list"):
                shot_templates = {
                    "hook": ("close", "fast push-in", "HOOK"),
                    "problem": ("medium", "slow push-in", "THE PROBLEM"),
                    "reframe": ("medium", "subtle pan", "THE SHIFT"),
                    "solution": ("wide", "gimbal forward", "HOW IT WORKS"),
                    "payoff": ("close", "soft drift", "THE RESULT"),
                    "cta": ("medium", "locked center", "DOWNLOAD NOW"),
                    "auto": ("medium", "slow push-in", "NEXT BEAT"),
                }
                shot_type_a, cam_a, text_a = shot_templates.get(
                    scene_type, shot_templates["auto"]
                )
                text_b = short_goal if short_goal else "CONTINUE"
                scene["shot_list"] = [
                    {
                        "shot_type": shot_type_a,
                        "camera_move": cam_a,
                        "subject_focus": "primary speaker",
                        "duration_seconds": 2.8,
                        "on_screen_text": text_a,
                        "music_mood": "uplifting",
                        "sound_effects": ["soft whoosh"],
                        "transition_out": "cut" if scene_type in {"hook", "problem"} else "fade",
                    },
                    {
                        "shot_type": "close",
                        "camera_move": "subtle pan" if scene_type != "cta" else "gentle zoom-in",
                        "subject_focus": "reaction and CTA",
                        "duration_seconds": 3.0,
                        "on_screen_text": text_b,
                        "music_mood": "uplifting",
                        "sound_effects": ["click accent"],
                        "transition_out": "fade",
                    },
                ]
                self._record(
                    manifest,
                    f"scenes.{scene.get('scene_id')}.shot_list",
                    scene["shot_list"],
                    reason,
                    seed_fragment,
                )

            for line in scene.get("dialogue", []):
                timeline_entries.append(
                    {
                        "scene_id": scene.get("scene_id"),
                        "speaker_id": line.get("speaker_id"),
                        "line": line.get("line"),
                        "start_second": line.get("start_second"),
                        "end_second": line.get("end_second"),
                    }
                )
            total_time += self._scene_duration(scene)

        transitions = []
        for idx in range(len(scenes) - 1):
            from_scene = scenes[idx].get("scene_id")
            to_scene = scenes[idx + 1].get("scene_id")
            transition = "cut"
            shot_list = scenes[idx].get("shot_list", [])
            if shot_list:
                transition = shot_list[-1].get("transition_out", "cut")
            transitions.append(
                {
                    "from_scene_id": from_scene,
                    "to_scene_id": to_scene,
                    "type": transition,
                    "duration_seconds": 0.4 if transition == "fade" else 0.0,
                }
            )

        manifest["timeline"] = {
            "entries": timeline_entries,
            "scene_order": [scene.get("scene_id") for scene in scenes if scene.get("scene_id")],
            "transitions": transitions,
            "pacing": {
                scene.get("scene_id"): 1.0 for scene in scenes if scene.get("scene_id")
            },
            "b_roll": manifest.get("timeline", {}).get("b_roll", [])
            if isinstance(manifest.get("timeline"), dict)
            else [],
            "scene_trims": manifest.get("timeline", {}).get("scene_trims", {})
            if isinstance(manifest.get("timeline"), dict)
            else {},
        }
        manifest.setdefault("approvals", {})
        manifest["approvals"]["script_approved"] = False
        manifest["approvals"]["script_needs_user_approval"] = True

    def _retime_scene_dialogue(
        self,
        scene: Dict[str, Any],
        scene_start: float,
    ) -> None:
        dialogue = scene.get("dialogue") or []
        if not isinstance(dialogue, list) or not dialogue:
            return

        cursor = float(scene_start)
        for line in dialogue:
            if not isinstance(line, dict):
                continue
            text = str(line.get("line", "")).strip()
            word_count = max(1, len(text.split()))
            duration = 0.0
            try:
                start = float(line.get("start_second", cursor))
                end = float(line.get("end_second", start))
                duration = end - start
            except Exception:
                duration = 0.0
            if duration <= 0:
                duration = max(1.2, min(4.5, word_count * 0.34))

            line["start_second"] = round(cursor, 2)
            line["end_second"] = round(cursor + duration, 2)
            cursor += duration

    def _fill_audio(
        self,
        manifest: Dict[str, Any],
        rng: random.Random,
        seed_fragment: str,
        reason: str,
        force: bool = False,
    ) -> None:
        audio = manifest.setdefault("audio", {})
        brief = manifest.get("brief", {})
        platform = str(brief.get("platform", "")).lower()

        music_default = "uplifting"
        if "linkedin" in platform:
            music_default = "corporate clean"
        elif "youtube" in platform:
            music_default = "cinematic light"
        elif "tiktok" in platform or "instagram" in platform:
            music_default = "energetic pop"

        if force or not audio.get("music_mood"):
            audio["music_mood"] = music_default
            self._record(manifest, "audio.music_mood", music_default, reason, seed_fragment)

        if force or not audio.get("sound_effects"):
            sfx = ["soft whoosh", "button click"]
            audio["sound_effects"] = sfx
            self._record(manifest, "audio.sound_effects", sfx, reason, seed_fragment)

    def _fill_output(
        self,
        manifest: Dict[str, Any],
        rng: random.Random,
        seed_fragment: str,
        reason: str,
        force: bool = False,
    ) -> None:
        output = manifest.setdefault("output", {})
        brief = manifest.get("brief", {})

        defaults = {
            "resolution": "1080p",
            "fps": 30,
            "subtitles_enabled": True,
            "subtitle_style": "clean lower-third",
            "safe_areas": "social-safe margins",
            "file_format": "mp4",
            "deliverables": ["final_video", "script", "scene_list", "assets_list"],
        }

        platform = str(brief.get("platform", "")).lower()
        if "youtube" in platform:
            defaults["aspect_note"] = "safe for 16:9 overlays"
        elif "tiktok" in platform or "instagram" in platform:
            defaults["aspect_note"] = "safe for vertical UI overlays"
        else:
            defaults["aspect_note"] = "safe for multi-platform overlays"

        for field, value in defaults.items():
            if force or output.get(field) in (None, "", []):
                output[field] = value
                self._record(manifest, f"output.{field}", value, reason, seed_fragment)

    def sync_character_bible(self, manifest: Dict[str, Any], reason: str = "sync") -> None:
        bible = manifest.setdefault("character_bible", {})
        world = manifest.get("world", {})
        for character in manifest.get("characters", []):
            char_id = character.get("character_id")
            if not char_id:
                continue
            entry = bible.setdefault(
                char_id,
                {
                    "canonical_appearance": "",
                    "wardrobe_lock": "",
                    "props_lock": "",
                    "continuity_notes": "",
                    "reference_images": [],
                    "visual_prompt_tokens": [],
                },
            )
            char_refs = [
                item
                for item in character.get("reference_images", [])
                if isinstance(item, str) and item.strip()
            ]
            entry_refs = entry.setdefault("reference_images", [])
            for ref in char_refs:
                if ref not in entry_refs:
                    entry_refs.append(ref)
            has_refs = bool(entry_refs)

            canonical = str(entry.get("canonical_appearance", "") or "").strip()
            if (not canonical) or (has_refs and "reference image" not in canonical.lower()):
                if has_refs:
                    canonical = (
                        "Match uploaded reference images for face identity, skin tone, body proportions, and hairstyle."
                    )
                else:
                    canonical = character.get("appearance", "")
                entry["canonical_appearance"] = canonical
                self._record(
                    manifest,
                    f"character_bible.{char_id}.canonical_appearance",
                    entry["canonical_appearance"],
                    reason,
                    "sync",
                )

            wardrobe_lock = str(entry.get("wardrobe_lock", "") or "").strip()
            world_wardrobe = str(world.get("wardrobe_props", "") or "").strip()
            if (not wardrobe_lock) or (has_refs and wardrobe_lock == world_wardrobe):
                if has_refs:
                    wardrobe_lock = (
                        "Lock wardrobe to uploaded reference images unless a scene explicitly requests a wardrobe change."
                    )
                else:
                    wardrobe_lock = world_wardrobe
                entry["wardrobe_lock"] = wardrobe_lock

            if not entry.get("props_lock"):
                entry["props_lock"] = ""

            if not entry.get("continuity_notes"):
                if has_refs:
                    entry["continuity_notes"] = (
                        "Treat character reference images as hard continuity constraints for every scene."
                    )
                else:
                    entry["continuity_notes"] = (
                        "Keep facial features, hairstyle, and wardrobe consistent across scenes."
                    )
            if not entry.get("visual_prompt_tokens"):
                entry["visual_prompt_tokens"] = [
                    character.get("name", char_id),
                    character.get("personality", ""),
                ]

    def regenerate_scene_content(
        self,
        manifest: Dict[str, Any],
        scene_id: str,
        *,
        branch_index: int = 0,
    ) -> Dict[str, Any]:
        scenes = manifest.get("scenes", [])
        target = None
        target_index = -1
        for idx, scene in enumerate(scenes):
            if scene.get("scene_id") == scene_id:
                target = scene
                target_index = idx
                break
        if target is None:
            raise ValueError(f"Scene '{scene_id}' not found.")

        rng, seed_fragment = self._rng_for_step(
            manifest, f"scene_regen:{scene_id}", surprise_index=branch_index
        )

        scene_copy = {
            **target,
            "events": list(target.get("events", [])),
            "dialogue": list(target.get("dialogue", [])),
            "shot_list": list(target.get("shot_list", [])),
        }

        scene_copy["events"] = [
            "Hook beat: alternate opening with stronger pattern interrupt.",
            "Middle beat: revised proof moment with character reaction.",
            "End beat: revised CTA tailored to audience urgency.",
        ]
        hook_options = [
            "This is the upgraded version you asked for.",
            "Here is the revised cut with stronger impact.",
            "Let us run a better take for this scene.",
        ]
        payoff_options = [
            "Use this version and keep the momentum going.",
            "Confirm this take and continue generation.",
            "Approve this scene to carry continuity forward.",
        ]
        first_speaker = None
        second_speaker = None
        chars = scene_copy.get("characters_in_scene", [])
        if chars:
            first_speaker = chars[0]
            second_speaker = chars[1] if len(chars) > 1 else chars[0]
        else:
            first_speaker = "char_1"
            second_speaker = "char_2"

        start = 0.0
        if scene_copy.get("dialogue"):
            start = float(scene_copy["dialogue"][0].get("start_second", 0.0))

        scene_copy["dialogue"] = [
            {
                "speaker_id": first_speaker,
                "line": rng.choice(hook_options),
                "emotion": "confident",
                "pause_notes": "brief beat",
                "start_second": round(start, 2),
                "end_second": round(start + 2.9, 2),
            },
            {
                "speaker_id": second_speaker,
                "line": rng.choice(payoff_options),
                "emotion": "assured",
                "pause_notes": "end with validation cue",
                "start_second": round(start + 2.9, 2),
                "end_second": round(start + 5.9, 2),
            },
        ]

        scene_copy["shot_list"] = [
            {
                "shot_type": "medium",
                "camera_move": "fast push-in",
                "subject_focus": "primary speaker",
                "duration_seconds": 2.9,
                "on_screen_text": "Revised Hook",
                "music_mood": "uplifting",
                "sound_effects": ["impact whoosh"],
                "transition_out": "cut",
            },
            {
                "shot_type": "close",
                "camera_move": "rack focus",
                "subject_focus": "CTA expression",
                "duration_seconds": 3.0,
                "on_screen_text": "Approve or edit",
                "music_mood": "uplifting",
                "sound_effects": ["soft hit"],
                "transition_out": "fade",
            },
        ]
        scene_copy["variant_seed"] = seed_fragment
        scene_copy["locked"] = False
        scene_copy["state"] = "planned"

        scenes[target_index] = scene_copy
        manifest["scenes"] = scenes
        self._fill_script(manifest, rng, seed_fragment, "scene_regenerate", replace=False)
        self._record(
            manifest,
            f"scenes.{scene_id}.regenerated",
            scene_copy,
            "scene_regenerate",
            seed_fragment,
        )
        manifest.setdefault("approvals", {})
        manifest["approvals"]["script_approved"] = False
        manifest["approvals"]["script_needs_user_approval"] = True
        return scene_copy

    def _sync_scene_characters_with_cast(self, manifest: Dict[str, Any]) -> None:
        characters = [
            item
            for item in manifest.get("characters", [])
            if isinstance(item, dict) and item.get("character_id")
        ]
        if not characters:
            return

        valid_ids = [item.get("character_id") for item in characters if item.get("character_id")]
        alias_to_id: Dict[str, str] = {}
        for character in characters:
            char_id = str(character.get("character_id", "")).strip()
            if not char_id:
                continue
            alias_to_id[char_id.lower()] = char_id
            name = str(character.get("name", "")).strip().lower()
            if name:
                alias_to_id[name] = char_id
                alias_to_id[name.replace(" ", "_")] = char_id
            alias_to_id[char_id.replace("char_", "").lower()] = char_id

        for scene in manifest.get("scenes", []):
            if not isinstance(scene, dict):
                continue

            current_scene_ids = []
            for raw in scene.get("characters_in_scene", []) or []:
                key = str(raw or "").strip().lower()
                mapped = alias_to_id.get(key)
                if mapped and mapped not in current_scene_ids:
                    current_scene_ids.append(mapped)
            if not current_scene_ids:
                current_scene_ids = valid_ids[:2]
            scene["characters_in_scene"] = current_scene_ids[:2]

            for line in scene.get("dialogue", []) or []:
                if not isinstance(line, dict):
                    continue
                speaker_raw = str(line.get("speaker_id", "")).strip()
                if not speaker_raw:
                    line["speaker_id"] = current_scene_ids[0]
                    continue
                speaker_key = speaker_raw.lower()
                mapped = alias_to_id.get(speaker_key)
                if mapped:
                    line["speaker_id"] = mapped
                    continue
                if speaker_key in {"narrator", "voiceover", "on-screen_text"}:
                    continue
                if speaker_key not in {item.lower() for item in valid_ids}:
                    line["speaker_id"] = current_scene_ids[0]

    @staticmethod
    def _short_scene_goal(goal: str, max_words: int = 6) -> str:
        words = [word.strip(".,:;!?") for word in str(goal or "").split() if word.strip()]
        if not words:
            return ""
        snippet = " ".join(words[:max_words])
        return snippet.upper()

    def _record(
        self,
        manifest: Dict[str, Any],
        field_path: str,
        value: Any,
        reason: str,
        seed_fragment: str,
    ) -> None:
        log = manifest.setdefault("autofill_log", [])
        log.append(
            {
                "field_path": field_path,
                "value": value,
                "source": "autofill",
                "reason": reason,
                "generator_version": "v2.0",
                "seed_fragment": seed_fragment,
                "timestamp": utc_now_iso(),
            }
        )

    def _build_logline(self, brief: Dict[str, Any], rng: random.Random) -> str:
        tone = brief.get("tone") or "engaging"
        goal = brief.get("goal") or "encourage action"
        audience = brief.get("target_audience") or "viewers"
        templates = [
            f"A {tone} short where a relatable character discovers a fast win and nudges {audience} to {goal.lower()}",
            f"{audience.title()} see a clear before/after moment that drives them to {goal.lower()}",
            f"A punchy narrative that turns curiosity into action for {audience}",
        ]
        return rng.choice(templates)

    def _scene_goal(self, brief: Dict[str, Any], index: int) -> str:
        goal = str(brief.get("goal", "drive action"))
        sequence = [
            f"Hook viewer attention in first 1-2 seconds and frame the promise: {goal}",
            f"Show the core problem or friction blocking the viewer from: {goal}",
            f"Reframe the problem with a simpler, low-pressure path toward: {goal}",
            f"Demonstrate the solution steps that move the viewer toward: {goal}",
            f"Deliver emotional payoff and proof that this works for the audience.",
            f"Close with a direct CTA tied to the goal: {goal}",
        ]
        if 0 <= index < len(sequence):
            return sequence[index]
        return f"Advance the narrative while reinforcing the goal: {goal}"

    @staticmethod
    def _default_scene_type(index: int) -> str:
        sequence = ["hook", "problem", "reframe", "solution", "payoff", "cta"]
        if 0 <= index < len(sequence):
            return sequence[index]
        return "custom"

    def _scene_duration(self, scene: Dict[str, Any]) -> float:
        shots = scene.get("shot_list") or []
        if shots:
            return float(sum(float(shot.get("duration_seconds", 0)) for shot in shots))
        dialogue = scene.get("dialogue") or []
        if dialogue:
            start = float(dialogue[0].get("start_second", 0))
            end = float(dialogue[-1].get("end_second", start + 6))
            return max(1.0, end - start)
        return 6.0
