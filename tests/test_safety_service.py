import unittest

from services.safety_service import SafetyCheckService


class SafetyCheckTests(unittest.TestCase):
    def test_detects_warning_terms(self):
        service = SafetyCheckService()
        manifest = {
            "brief": {"goal": "clone voice exactly like a famous person"},
            "scenes": [
                {
                    "scene_id": "scene_001",
                    "dialogue": [{"line": "This includes Nike branding and violence"}],
                }
            ],
        }
        warnings = service.run_checks(manifest)
        categories = {item["category"] for item in warnings}
        self.assertIn("impersonation", categories)
        self.assertIn("trademark", categories)
        self.assertIn("unsafe_content", categories)


if __name__ == "__main__":
    unittest.main()
