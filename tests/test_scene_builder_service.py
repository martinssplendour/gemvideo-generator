import random
import unittest

from services.scene_builder_service import SceneBuilderService


class _StubModels:
    def __init__(self, text: str):
        self._text = text

    def generate_content(self, **kwargs):
        class _Resp:
            pass

        resp = _Resp()
        resp.text = self._text
        return resp


class _StubClient:
    def __init__(self, text: str):
        self.models = _StubModels(text)


class SceneBuilderServiceTests(unittest.TestCase):
    def test_model_plan_is_applied_when_enabled(self):
        model_json = """
        {
          "scenes": [
            {
              "scene_id": "scene_001",
              "scene_type": "hook",
              "scene_goal": "Open with Sue speaking directly to new-city introverts.",
              "location_time": "Apartment, evening",
              "characters_in_scene": ["Sue"],
              "events": ["Hook in 1 second", "State core pain", "Promise easy social entry"],
              "dialogue": [
                {"speaker_id": "Sue", "line": "If making friends feels awkward, this is for you.", "duration_seconds": 3.0}
              ],
              "shot_list": [
                {"shot_type": "close", "camera_move": "handheld", "subject_focus": "Sue", "duration_seconds": 3.0}
              ]
            },
            {
              "scene_id": "scene_002",
              "scene_type": "solution",
              "scene_goal": "Show swipe-to-join flow with social plans.",
              "location_time": "City street, night",
              "characters_in_scene": ["char_sue"],
              "events": ["Show app interaction", "Show group activity", "Lead into CTA"]
            }
          ]
        }
        """
        service = SceneBuilderService(
            api_key="real_key",
            model_name="gemini-test",
            enabled=True,
            client=_StubClient(model_json),
        )
        manifest = {
            "project_seed": "seed_123",
            "brief": {
                "goal": "Drive installs",
                "platform": "TikTok",
                "duration_seconds": 16,
                "tone": "warm",
            },
            "world": {"setting": "City", "camera_style": "handheld", "lighting": "natural"},
            "characters": [
                {"character_id": "char_sue", "name": "Sue", "role": "protagonist"}
            ],
            "scenes": [],
        }

        scenes = service.build_scenes(manifest, random.Random(7), replace=True)

        self.assertEqual(len(scenes), 2)
        self.assertEqual(scenes[0]["scene_goal"], "Open with Sue speaking directly to new-city introverts.")
        self.assertEqual(
            scenes[0]["scene_prompt"],
            "Open with Sue speaking directly to new-city introverts.",
        )
        self.assertTrue(scenes[0]["conversation"])
        self.assertEqual(scenes[0]["characters_in_scene"], ["char_sue"])
        self.assertTrue(scenes[0]["builder"]["model_used"])
        self.assertEqual(scenes[1]["continuity"]["from_scene_id"], "scene_001")
        self.assertTrue(scenes[1]["continuity"]["must_follow_previous_frame"])

    def test_build_scenes_requires_gemini_when_model_disabled(self):
        service = SceneBuilderService(
            api_key="dummy_key",
            model_name="gemini-test",
            enabled=True,
            client=_StubClient('{"scenes":[{"scene_id":"scene_001","scene_goal":"Ignored"}]}'),
        )
        manifest = {
            "project_seed": "seed_123",
            "brief": {
                "goal": "Drive installs",
                "platform": "TikTok",
                "duration_seconds": 8,
                "tone": "warm",
                "scene_count_hint": 1,
            },
            "world": {"setting": "City", "camera_style": "handheld", "lighting": "natural"},
            "characters": [
                {"character_id": "char_sue", "name": "Sue", "role": "protagonist"}
            ],
            "scenes": [],
        }

        with self.assertRaisesRegex(RuntimeError, "requires Gemini"):
            service.build_scenes(manifest, random.Random(7), replace=True)

    def test_non_replace_keeps_existing_scenes_when_model_returns_no_plan(self):
        service = SceneBuilderService(
            api_key="real_key",
            model_name="gemini-test",
            enabled=True,
            client=_StubClient("{}"),
        )
        manifest = {
            "project_seed": "seed_123",
            "brief": {
                "goal": "Drive installs",
                "platform": "TikTok",
                "duration_seconds": 8,
                "tone": "warm",
                "scene_count_hint": 1,
            },
            "world": {"setting": "City", "camera_style": "handheld", "lighting": "natural"},
            "characters": [
                {"character_id": "char_sue", "name": "Sue", "role": "protagonist"}
            ],
            "scenes": [
                {
                    "scene_id": "scene_001",
                    "scene_type": "hook",
                    "scene_prompt": "Existing hook line.",
                    "scene_goal": "Existing hook line.",
                    "conversation": ["char_sue: Existing line."],
                    "characters_in_scene": ["char_sue"],
                }
            ],
        }

        scenes = service.build_scenes(manifest, random.Random(7), replace=False)

        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0]["scene_prompt"], "Existing hook line.")
        self.assertFalse(scenes[0]["builder"]["model_used"])

    def test_model_plan_accepts_scene_list_alias(self):
        model_json = """
        {
          "scene_list": [
            {
              "scene_id": "scene_001",
              "scene_type": "hook",
              "scene_prompt": "Open with a tight emotional hook.",
              "conversation": ["char_sue: I am done settling."]
            }
          ]
        }
        """
        service = SceneBuilderService(
            api_key="real_key",
            model_name="gemini-test",
            enabled=True,
            client=_StubClient(model_json),
        )
        manifest = {
            "project_seed": "seed_123",
            "brief": {
                "goal": "Drive installs",
                "platform": "TikTok",
                "duration_seconds": 8,
                "tone": "warm",
                "scene_count_hint": 1,
            },
            "world": {"setting": "City", "camera_style": "handheld", "lighting": "natural"},
            "characters": [
                {"character_id": "char_sue", "name": "Sue", "role": "protagonist"}
            ],
            "scenes": [],
        }

        scenes = service.build_scenes(manifest, random.Random(7), replace=True)

        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0]["scene_prompt"], "Open with a tight emotional hook.")

    def test_model_scene_goal_and_description_are_preserved(self):
        model_json = """
        {
          "scenes": [
            {
              "scene_id": "scene_001",
              "scene_type": "problem",
              "scene_prompt": "A woman is asked to split an expensive date bill and freezes in disbelief.",
              "scene_goal": "Establish the unfair-date pain point with clear emotional impact.",
              "scene_description": "Restaurant table, card machine in frame, tight reaction shot, awkward silence.",
              "conversation": ["char_woman: Wait, you invited me and still want me to split this?"]
            },
            {
              "scene_id": "scene_002",
              "scene_type": "solution",
              "scene_prompt": "Narrator introduces Splendoure as the way to set date expectations before matching.",
              "scene_goal": "Resolve both frustrations by presenting Splendoure as the clarity-first filter.",
              "scene_description": "Phone UI demo with clear date-plan preferences and mutual agreement.",
              "conversation": ["char_narrator: Set what you want upfront. Only matching expectations swipe in."]
            }
          ]
        }
        """
        service = SceneBuilderService(
            api_key="real_key",
            model_name="gemini-test",
            enabled=True,
            client=_StubClient(model_json),
        )
        manifest = {
            "project_seed": "seed_456",
            "brief": {
                "description": "Two people had bad date outcomes and narrator introduces Splendoure.",
                "goal": "Drive installs",
                "platform": "TikTok",
                "duration_seconds": 16,
                "scene_count_hint": 2,
            },
            "world": {"setting": "City", "camera_style": "handheld", "lighting": "natural"},
            "characters": [
                {"character_id": "char_woman", "name": "Woman", "role": "protagonist"},
                {"character_id": "char_narrator", "name": "Narrator", "role": "narrator"},
            ],
            "scenes": [],
        }

        scenes = service.build_scenes(manifest, random.Random(7), replace=True)

        self.assertEqual(len(scenes), 2)
        self.assertIn("pain point", scenes[0]["scene_goal"])
        self.assertIn("card machine", scenes[0]["scene_description"])
        self.assertIn("Splendoure", scenes[1]["scene_goal"])
        self.assertIn("UI demo", scenes[1]["scene_description"])


if __name__ == "__main__":
    unittest.main()
