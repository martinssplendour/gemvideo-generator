import copy
import unittest

from services.autofill_service import ManifestAutoFillService


class _RecordingSceneBuilder:
    def __init__(self):
        self.replace_calls = []

    def build_scenes(self, manifest, rng, *, replace):
        self.replace_calls.append(bool(replace))
        return [
            {
                "scene_id": "scene_001",
                "scene_goal": "Model-first scene",
                "scene_type": "hook",
                "locked": False,
                "state": "planned",
                "characters_in_scene": ["char_sue"],
                "events": ["model event"],
                "dialogue": [],
                "shot_list": [],
            }
        ]


class _TestSceneBuilder:
    def build_scenes(self, manifest, rng, *, replace):
        del rng
        duration = int(manifest.get("brief", {}).get("duration_seconds", 15) or 15)
        scene_count = max(1, round(duration / 8))
        characters = [
            item
            for item in manifest.get("characters", [])
            if isinstance(item, dict) and item.get("character_id")
        ]
        cast_ids = [item["character_id"] for item in characters]
        if not cast_ids:
            cast_ids = ["char_1"]
        existing_map = {
            item.get("scene_id"): item
            for item in manifest.get("scenes", [])
            if isinstance(item, dict) and item.get("scene_id")
        }

        scenes = []
        previous_scene_id = ""
        for index in range(scene_count):
            scene_id = f"scene_{index + 1:03d}"
            existing = existing_map.get(scene_id)
            if isinstance(existing, dict) and existing.get("locked"):
                scene = dict(existing)
            else:
                label = "Surprise" if replace else "Planned"
                scene = {
                    "scene_id": scene_id,
                    "scene_goal": f"{label} scene {index + 1}",
                    "scene_type": "hook" if index == 0 else "problem",
                    "scene_description": f"{label} description {index + 1}",
                    "spoken_direction": f"{label} spoken direction {index + 1}",
                    "image_usage_instructions": f"{label} image instructions {index + 1}",
                    "locked": False,
                    "state": "planned",
                    "characters_in_scene": cast_ids[:2],
                    "events": [f"{label.lower()} event {index + 1}"],
                    "dialogue": [],
                    "shot_list": [],
                }

            scene["characters_in_scene"] = scene.get("characters_in_scene") or cast_ids[:2]
            scene["continuity"] = {
                "scene_id": scene_id,
                "chain_index": index + 1,
                "from_scene_id": previous_scene_id or None,
                "must_follow_previous_frame": bool(previous_scene_id),
                "opening_anchor": (
                    "previous_scene_end_frame"
                    if previous_scene_id
                    else "character_reference_images"
                ),
                "strategy": "last_frame_plus_next_goal",
                "character_reference_ids": list(scene["characters_in_scene"]),
                "style_lock": {"tone": "", "camera_style": "", "lighting": ""},
                "scene_type": scene.get("scene_type", "hook"),
            }
            if scene.get("locked"):
                scene["state"] = "locked"
            scenes.append(scene)
            previous_scene_id = scene_id
        return scenes


class AutoFillDeterminismTests(unittest.TestCase):
    def setUp(self):
        self.service = ManifestAutoFillService(scene_builder_service=_TestSceneBuilder())
        self.base_manifest = {
            "project_id": "proj1",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "project_seed": "seed-123",
            "surprise_counters": {},
            "brief": {
                "video_type": "social skit",
                "goal": "Drive signups",
                "target_audience": "creators",
                "platform": "TikTok",
                "duration_seconds": 15,
                "aspect_ratio": "9:16",
                "tone": "funny",
                "language": "English",
                "accent": "neutral",
            },
            "characters": [],
            "world": {},
            "scenes": [],
            "audio": {},
            "output": {},
            "timeline": [],
            "autofill_log": [],
        }

    def test_autofill_all_is_deterministic_for_same_seed(self):
        a = copy.deepcopy(self.base_manifest)
        b = copy.deepcopy(self.base_manifest)

        self.service.autofill_all(a, reason="test")
        self.service.autofill_all(b, reason="test")

        self.assertEqual(a["characters"], b["characters"])
        self.assertEqual(a["world"], b["world"])
        self.assertEqual(a["scenes"], b["scenes"])
        self.assertEqual(a["output"], b["output"])

    def test_surprise_changes_scene_content(self):
        manifest = copy.deepcopy(self.base_manifest)
        self.service.autofill_step(manifest, "scenes", reason="init")
        first = copy.deepcopy(manifest["scenes"])

        self.service.surprise_step(manifest, "scenes")
        second = manifest["scenes"]
        self.assertNotEqual(first, second)

    def test_scene_builder_populates_continuity_chain(self):
        manifest = copy.deepcopy(self.base_manifest)
        manifest["brief"]["duration_seconds"] = 24
        manifest["characters"] = [
            {
                "character_id": "char_sue",
                "name": "Sue",
                "role": "protagonist",
                "appearance": "Age 24-30, casual",
                "personality": "calm",
                "voice": {"voice_provider": "native", "voice_id": "native_sue"},
            }
        ]

        self.service.autofill_step(manifest, "scenes", reason="init")
        scenes = manifest["scenes"]

        self.assertGreaterEqual(len(scenes), 3)
        first = scenes[0]
        second = scenes[1]

        self.assertEqual(first["characters_in_scene"], ["char_sue"])
        self.assertIn("continuity", first)
        self.assertFalse(first["continuity"]["must_follow_previous_frame"])
        self.assertEqual(first["continuity"]["opening_anchor"], "character_reference_images")
        self.assertTrue(first.get("scene_description"))
        self.assertTrue(first.get("spoken_direction"))
        self.assertTrue(first.get("image_usage_instructions"))

        self.assertTrue(second["continuity"]["must_follow_previous_frame"])
        self.assertEqual(second["continuity"]["from_scene_id"], "scene_001")
        self.assertEqual(second["characters_in_scene"], ["char_sue"])

    def test_scene_builder_preserves_locked_scene_on_surprise(self):
        manifest = copy.deepcopy(self.base_manifest)
        manifest["brief"]["duration_seconds"] = 24
        manifest["characters"] = [
            {
                "character_id": "char_sue",
                "name": "Sue",
                "role": "protagonist",
                "appearance": "Age 24-30, casual",
                "personality": "calm",
                "voice": {"voice_provider": "native", "voice_id": "native_sue"},
            }
        ]
        manifest["scenes"] = [
            {
                "scene_id": "scene_001",
                "scene_goal": "Manual locked hook",
                "scene_type": "hook",
                "locked": True,
                "state": "locked",
                "characters_in_scene": ["char_sue"],
                "events": ["manual event"],
                "dialogue": [],
                "shot_list": [],
            }
        ]

        self.service.surprise_step(manifest, "scenes")
        scenes = manifest["scenes"]
        first = scenes[0]
        self.assertTrue(first["locked"])
        self.assertEqual(first["scene_goal"], "Manual locked hook")
        self.assertIn("continuity", first)
        self.assertEqual(first["continuity"]["chain_index"], 1)

    def test_force_scene_rebuild_uses_replace_mode(self):
        scene_builder = _RecordingSceneBuilder()
        service = ManifestAutoFillService(scene_builder_service=scene_builder)
        manifest = copy.deepcopy(self.base_manifest)
        manifest["characters"] = [
            {
                "character_id": "char_sue",
                "name": "Sue",
                "role": "protagonist",
                "appearance": "Age 24-30, casual",
                "personality": "calm",
                "voice": {"voice_provider": "native", "voice_id": "native_sue"},
            }
        ]
        manifest["scenes"] = [
            {
                "scene_id": "scene_001",
                "scene_goal": "Placeholder scene",
                "scene_type": "hook",
                "locked": False,
                "state": "planned",
                "characters_in_scene": ["char_sue"],
                "events": [],
                "dialogue": [],
                "shot_list": [],
            }
        ]

        service.autofill_all(manifest, reason="test", force_scene_rebuild=True)

        self.assertEqual(scene_builder.replace_calls, [True])
        self.assertEqual(manifest["scenes"][0]["scene_goal"], "Model-first scene")


if __name__ == "__main__":
    unittest.main()
