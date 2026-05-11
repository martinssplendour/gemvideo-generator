import unittest

from services.description_planner_service import DescriptionPlannerService


class _StubModels:
    def __init__(self, text):
        self._text = text

    def generate_content(self, **kwargs):
        class _Resp:
            pass

        resp = _Resp()
        resp.text = self._text
        return resp


class _StubClient:
    def __init__(self, text):
        self.models = _StubModels(text)


class DescriptionPlannerServiceTests(unittest.TestCase):
    def test_plan_manifest_patch_parses_and_sanitizes_json(self):
        client = _StubClient(
            """```json
            {
              "brief": {
                "video_type": "ad",
                "platform": "TikTok",
                "duration_seconds": 16,
                "language": "English"
              },
              "characters": [
                {
                  "name": "Ava",
                  "role": "host",
                  "appearance": "Red jacket",
                  "personality": "confident"
                }
              ],
              "scene_plan": [
                {"scene_goal": "Hook fast"},
                {"scene_goal": "Show result"},
                {"scene_goal": "Strong CTA"}
              ],
              "script": [
                {"speaker": "ava", "line": "Try this now."}
              ],
              "assumptions": ["Defaulted to vertical for TikTok."]
            }
            ```"""
        )
        service = DescriptionPlannerService(
            api_key="real_key",
            model_name="gemini-2.0-flash",
            enabled=True,
            client=client,
        )
        patch = service.plan_manifest_patch("Make a quick TikTok ad.")
        self.assertEqual(patch["brief"]["platform"], "TikTok")
        self.assertEqual(patch["characters"][0]["name"], "Ava")
        self.assertTrue(patch["scene_plan"])
        self.assertEqual(patch["scene_plan"][0]["scene_goal"], "Hook fast")
        self.assertIn("Hook fast", patch["scene_plan_text"])
        self.assertIn("ava: Try this now.", patch["script_text"])
        self.assertTrue(patch["assumptions"])

    def test_disabled_service_returns_empty_patch(self):
        service = DescriptionPlannerService(
            api_key="dummy_key",
            model_name="gemini-2.0-flash",
            enabled=True,
            client=_StubClient("{}"),
        )
        self.assertEqual(service.plan_manifest_patch("some description"), {})


if __name__ == "__main__":
    unittest.main()
