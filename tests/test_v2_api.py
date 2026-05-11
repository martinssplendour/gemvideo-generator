import io
import json
import os
import re
import shutil
import tempfile
import unittest
from unittest.mock import patch

from services.simple_video_service import SimpleVideoResult


class _StubGenAIModels:
    def generate_content(self, **kwargs):
        contents = str(kwargs.get("contents", "") or "")
        response = type("_Resp", (), {})()
        if "Return strict JSON only with this shape" not in contents:
            response.text = "{}"
            return response

        match = re.search(r"Return exactly (\d+) scenes\.", contents)
        scene_count = max(1, int(match.group(1))) if match else 2
        scenes = []
        for index in range(scene_count):
            scene_id = f"scene_{index + 1:03d}"
            scenes.append(
                {
                    "scene_id": scene_id,
                    "scene_type": "hook" if index == 0 else "problem",
                    "scene_goal": f"Gemini scene {index + 1}",
                    "scene_description": f"Gemini-built scene {index + 1}",
                    "spoken_direction": f"Deliver scene {index + 1} clearly.",
                    "image_usage_instructions": "Use provided references consistently.",
                    "location_time": "City, evening",
                    "characters_in_scene": ["char_1"],
                    "events": [f"Beat {index + 1}A", f"Beat {index + 1}B"],
                    "dialogue": [],
                    "shot_list": [],
                    "reference_images": [],
                    "narration": "",
                }
            )
        response.text = json.dumps({"scenes": scenes})
        return response


class _StubGenAIClient:
    def __init__(self):
        self.models = _StubGenAIModels()


class V2ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="video_builder_v2_test_")
        os.environ["DATA_DIR"] = os.path.join(cls.temp_dir, "data")
        os.environ["OUTPUT_DIR"] = os.path.join(cls.temp_dir, "output")
        os.environ["UPLOAD_DIR"] = os.path.join(cls.temp_dir, "uploads")
        os.environ["ENABLE_REAL_GENERATION"] = "false"
        os.environ["ENABLE_MANIFEST_PLANNER"] = "false"
        os.environ["GEMINI_API_KEY"] = "scene_builder_test_key"
        cls.genai_patcher = patch("google.genai.Client", return_value=_StubGenAIClient())
        cls.genai_patcher.start()

        from application import create_app

        cls.app = create_app()
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls.genai_patcher.stop()
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_create_project_and_skip_steps(self):
        create_resp = self.client.post(
            "/api/v2/projects",
            json={
                "brief": {
                    "video_type": "ad",
                    "goal": "drive trial signups",
                    "target_audience": "small business owners",
                    "platform": "TikTok",
                    "duration_seconds": 15,
                    "aspect_ratio": "9:16",
                    "tone": "funny",
                    "language": "English",
                    "accent": "neutral",
                }
            },
        )
        self.assertEqual(create_resp.status_code, 201)
        data = create_resp.get_json()
        project_id = data["project_id"]

        skip_resp = self.client.post(f"/api/v2/projects/{project_id}/steps/scenes/skip")
        self.assertEqual(skip_resp.status_code, 200)
        skip_data = skip_resp.get_json()
        self.assertTrue(len(skip_data["manifest_preview"]["scenes"]) >= 1)

    def test_legacy_endpoint_kept(self):
        resp = self.client.post("/api/generate", data={})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.get_json())

    def test_render_enqueue(self):
        create_resp = self.client.post(
            "/api/v2/projects",
            json={
                "brief": {
                    "video_type": "ad",
                    "goal": "drive trial signups",
                    "target_audience": "small business owners",
                    "platform": "TikTok",
                    "duration_seconds": 15,
                    "aspect_ratio": "9:16",
                    "tone": "funny",
                    "language": "English",
                    "accent": "neutral",
                }
            },
        )
        project_id = create_resp.get_json()["project_id"]
        self.client.post(f"/api/v2/projects/{project_id}/steps/review/skip")
        self.client.post(
            f"/api/v2/projects/{project_id}/approvals/script",
            json={"approved": True},
        )

        render_resp = self.client.post(f"/api/v2/projects/{project_id}/render")
        self.assertEqual(render_resp.status_code, 200)
        render_data = render_resp.get_json()
        self.assertEqual(render_data["status"], "queued")
        self.assertIn("job_id", render_data)
        self.assertEqual(render_data.get("runner"), "local_thread")

    def test_render_requires_script_approval(self):
        create_resp = self.client.post(
            "/api/v2/projects",
            json={
                "brief": {
                    "video_type": "ad",
                    "goal": "drive trial signups",
                    "target_audience": "small business owners",
                    "platform": "TikTok",
                    "duration_seconds": 15,
                    "aspect_ratio": "9:16",
                    "tone": "funny",
                    "language": "English",
                    "accent": "neutral",
                }
            },
        )
        project_id = create_resp.get_json()["project_id"]
        self.client.post(f"/api/v2/projects/{project_id}/steps/review/skip")
        render_resp = self.client.post(f"/api/v2/projects/{project_id}/render")
        self.assertEqual(render_resp.status_code, 400)
        self.assertIn("error", render_resp.get_json())

    def test_job_control_endpoints(self):
        create_resp = self.client.post(
            "/api/v2/projects",
            json={
                "brief": {
                    "video_type": "ad",
                    "goal": "drive trial signups",
                    "target_audience": "small business owners",
                    "platform": "TikTok",
                    "duration_seconds": 15,
                    "aspect_ratio": "9:16",
                    "tone": "funny",
                    "language": "English",
                    "accent": "neutral",
                }
            },
        )
        project_id = create_resp.get_json()["project_id"]
        self.client.post(f"/api/v2/projects/{project_id}/steps/review/skip")
        self.client.post(
            f"/api/v2/projects/{project_id}/approvals/script",
            json={"approved": True},
        )
        render_resp = self.client.post(f"/api/v2/projects/{project_id}/render")
        job_id = render_resp.get_json()["job_id"]

        pause_resp = self.client.post(f"/api/v2/jobs/{job_id}/pause")
        self.assertEqual(pause_resp.status_code, 200)
        self.assertTrue(pause_resp.get_json()["pause_requested"])

        resume_resp = self.client.post(f"/api/v2/jobs/{job_id}/resume")
        self.assertEqual(resume_resp.status_code, 200)
        self.assertFalse(resume_resp.get_json()["pause_requested"])

        cancel_resp = self.client.post(f"/api/v2/jobs/{job_id}/cancel")
        self.assertEqual(cancel_resp.status_code, 200)
        self.assertTrue(cancel_resp.get_json()["cancel_requested"])

    def test_scene_regeneration_and_character_bible_endpoint(self):
        create_resp = self.client.post(
            "/api/v2/projects",
            json={
                "brief": {
                    "video_type": "story",
                    "goal": "boost signups",
                    "target_audience": "students",
                    "platform": "Instagram",
                    "duration_seconds": 15,
                    "aspect_ratio": "9:16",
                    "tone": "serious",
                    "language": "English",
                    "accent": "neutral",
                }
            },
        )
        project_id = create_resp.get_json()["project_id"]
        self.client.post(f"/api/v2/projects/{project_id}/steps/scenes/skip")
        self.client.post(f"/api/v2/projects/{project_id}/steps/script/skip")

        regen_resp = self.client.post(
            f"/api/v2/projects/{project_id}/scenes/scene_001/regenerate",
            json={"branch": True, "activate": False},
        )
        self.assertEqual(regen_resp.status_code, 200)
        self.assertIn("scene_variants", regen_resp.get_json()["manifest_preview"])

        bible_resp = self.client.patch(
            f"/api/v2/projects/{project_id}/character-bible",
            json={
                "character_bible": {
                    "char_1": {
                        "canonical_appearance": "Blue jacket, braided hair",
                        "continuity_notes": "Keep same wardrobe across scenes",
                    }
                }
            },
        )
        self.assertEqual(bible_resp.status_code, 200)
        self.assertIn("character_bible", bible_resp.get_json()["manifest_preview"])

    def test_resume_approves_generated_scene_when_waiting_for_review(self):
        create_resp = self.client.post(
            "/api/v2/projects",
            json={
                "brief": {
                    "video_type": "ad",
                    "goal": "drive installs",
                    "target_audience": "students",
                    "platform": "TikTok",
                    "duration_seconds": 15,
                    "aspect_ratio": "9:16",
                    "tone": "serious",
                    "language": "English",
                    "accent": "neutral",
                }
            },
        )
        self.assertEqual(create_resp.status_code, 201)
        project_id = create_resp.get_json()["project_id"]
        self.client.post(f"/api/v2/projects/{project_id}/steps/scenes/skip")
        self.client.patch(
            f"/api/v2/projects/{project_id}/generation-mode",
            json={"generation_mode": "directed"},
        )

        manifest_service = self.app.config["V2_SERVICES"].manifest_service
        manifest_service.set_scene_state(
            project_id,
            "scene_001",
            "generated",
            note="Prepared for resume test.",
            force=True,
        )
        job = manifest_service.create_job(project_id)
        job_id = job["job_id"]
        manifest_service.update_job(
            project_id,
            job_id,
            status="paused",
            current_stage="awaiting_scene_approval",
            current_scene_id="scene_001",
        )

        resume_resp = self.client.post(f"/api/v2/jobs/{job_id}/resume")
        self.assertEqual(resume_resp.status_code, 200)
        resume_job = resume_resp.get_json()
        self.assertEqual(resume_job["status"], "running")
        self.assertFalse(resume_job["pause_requested"])
        self.assertIn("Approved scene_001", resume_job["control_note"])

        manifest = manifest_service.get_manifest(project_id)
        self.assertEqual(manifest["scenes"][0]["state"], "approved")

    def test_generation_mode_scene_controls_and_subtitle_edit(self):
        create_resp = self.client.post(
            "/api/v2/projects",
            json={
                "brief": {
                    "video_type": "explainer",
                    "goal": "increase trials",
                    "target_audience": "founders",
                    "platform": "TikTok",
                    "duration_seconds": 16,
                    "aspect_ratio": "9:16",
                    "tone": "serious",
                    "language": "English",
                    "accent": "neutral",
                }
            },
        )
        project_id = create_resp.get_json()["project_id"]
        self.client.post(f"/api/v2/projects/{project_id}/steps/scenes/skip")
        self.client.post(f"/api/v2/projects/{project_id}/steps/script/skip")

        mode_resp = self.client.patch(
            f"/api/v2/projects/{project_id}/generation-mode",
            json={"generation_mode": "directed"},
        )
        self.assertEqual(mode_resp.status_code, 200)
        self.assertEqual(
            mode_resp.get_json()["manifest_preview"]["generation_mode"], "directed"
        )

        approve_resp = self.client.post(
            f"/api/v2/projects/{project_id}/scenes/scene_001/approve",
            json={"approved": True},
        )
        self.assertEqual(approve_resp.status_code, 200)
        first_scene = approve_resp.get_json()["manifest_preview"]["scenes"][0]
        self.assertEqual(first_scene["state"], "approved")

        trim_resp = self.client.patch(
            f"/api/v2/projects/{project_id}/scenes/scene_001/trim",
            json={"start_trim": 0.3, "end_trim": 0.2},
        )
        self.assertEqual(trim_resp.status_code, 200)
        trims = trim_resp.get_json()["manifest_preview"]["timeline"]["scene_trims"]
        self.assertIn("scene_001", trims)

        subtitle_resp = self.client.patch(
            f"/api/v2/projects/{project_id}/scenes/scene_001/dialogue/0",
            json={
                "line": "Updated subtitle text for QA",
                "start_second": 0.0,
                "end_second": 2.7,
            },
        )
        self.assertEqual(subtitle_resp.status_code, 200)
        updated_scene = subtitle_resp.get_json()["manifest_preview"]["scenes"][0]
        self.assertEqual(
            updated_scene["dialogue"][0]["line"], "Updated subtitle text for QA"
        )

    def test_scene_builder_scene_edit_endpoint(self):
        create_resp = self.client.post(
            "/api/v2/projects",
            json={
                "brief": {
                    "video_type": "ad",
                    "goal": "drive installs",
                    "target_audience": "students",
                    "platform": "TikTok",
                    "duration_seconds": 16,
                    "aspect_ratio": "9:16",
                    "tone": "serious",
                    "language": "English",
                    "accent": "neutral",
                }
            },
        )
        self.assertEqual(create_resp.status_code, 201)
        project_id = create_resp.get_json()["project_id"]
        self.client.post(f"/api/v2/projects/{project_id}/steps/scenes/skip")

        patch_resp = self.client.patch(
            f"/api/v2/projects/{project_id}/scenes/scene_001",
            json={
                "scene_goal": "Stronger hook for the first 2 seconds",
                "scene_description": "Close-up of host with direct eye contact and phone-camera framing.",
                "spoken_direction": "Calm but punchy opening line.",
                "image_usage_instructions": "Use continuity from prior frame when available.",
                "events": ["Hook in first second", "Show clear benefit"],
            },
        )
        self.assertEqual(patch_resp.status_code, 200)
        scene = patch_resp.get_json()["manifest_preview"]["scenes"][0]
        self.assertEqual(scene["scene_goal"], "Stronger hook for the first 2 seconds")
        self.assertIn("phone-camera", scene["scene_description"])
        self.assertEqual(scene["state"], "planned")

    def test_create_project_returns_502_when_scene_builder_fails(self):
        scene_builder = self.app.config["V2_SERVICES"].scene_builder_service
        with patch.object(
            scene_builder,
            "build_scenes",
            side_effect=RuntimeError("Gemini scene builder returned no usable scenes."),
        ):
            response = self.client.post(
                "/api/v2/projects",
                json={
                    "description": "Create a short ad with two scenes.",
                    "brief": {"description": "Create a short ad with two scenes."},
                },
            )
        self.assertEqual(response.status_code, 502)
        payload = response.get_json()
        self.assertIn("Gemini scene builder returned no usable scenes", payload["error"])

    def test_update_step_returns_502_when_scene_builder_fails(self):
        create_resp = self.client.post(
            "/api/v2/projects",
            json={
                "brief": {
                    "video_type": "ad",
                    "goal": "drive installs",
                    "target_audience": "students",
                    "platform": "TikTok",
                    "duration_seconds": 16,
                    "aspect_ratio": "9:16",
                    "tone": "serious",
                    "language": "English",
                    "accent": "neutral",
                }
            },
        )
        self.assertEqual(create_resp.status_code, 201)
        project_id = create_resp.get_json()["project_id"]

        scene_builder = self.app.config["V2_SERVICES"].scene_builder_service
        with patch.object(
            scene_builder,
            "build_scenes",
            side_effect=RuntimeError("Gemini scene builder returned no usable scenes."),
        ):
            response = self.client.patch(
                f"/api/v2/projects/{project_id}/steps/scenes",
                json={"scene_plan_text": "Scene 1: Hook with a direct opening line."},
            )
        self.assertEqual(response.status_code, 502)
        payload = response.get_json()
        self.assertIn("Gemini scene builder returned no usable scenes", payload["error"])

    def test_scene_regenerate_video_endpoint(self):
        create_resp = self.client.post(
            "/api/v2/projects",
            json={
                "brief": {
                    "video_type": "ad",
                    "goal": "drive installs",
                    "target_audience": "students",
                    "platform": "TikTok",
                    "duration_seconds": 16,
                    "aspect_ratio": "9:16",
                    "tone": "serious",
                    "language": "English",
                    "accent": "neutral",
                }
            },
        )
        self.assertEqual(create_resp.status_code, 201)
        project_id = create_resp.get_json()["project_id"]
        self.client.post(f"/api/v2/projects/{project_id}/steps/scenes/skip")

        regen_resp = self.client.post(
            f"/api/v2/projects/{project_id}/scenes/scene_001/regenerate-video"
        )
        self.assertEqual(regen_resp.status_code, 200)
        payload = regen_resp.get_json()
        self.assertEqual(payload["mode"], "scene_only")
        self.assertEqual(payload["scene_id"], "scene_001")
        self.assertIn("job_id", payload)

    def test_estimate_snapshot_rollback_and_export_bundle(self):
        create_resp = self.client.post(
            "/api/v2/projects",
            json={
                "brief": {
                    "video_type": "ad",
                    "goal": "collect leads",
                    "target_audience": "agency owners",
                    "platform": "Instagram",
                    "duration_seconds": 24,
                    "aspect_ratio": "9:16",
                    "tone": "funny",
                    "language": "English",
                    "accent": "neutral",
                }
            },
        )
        project_id = create_resp.get_json()["project_id"]
        self.client.post(f"/api/v2/projects/{project_id}/steps/review/skip")

        estimate_resp = self.client.get(f"/api/v2/projects/{project_id}/estimate")
        self.assertEqual(estimate_resp.status_code, 200)
        self.assertIn("estimated_api_cost_usd", estimate_resp.get_json())

        snapshot_resp = self.client.post(
            f"/api/v2/projects/{project_id}/snapshots",
            json={"label": "baseline"},
        )
        self.assertEqual(snapshot_resp.status_code, 201)
        snapshot_id = snapshot_resp.get_json()["snapshot_id"]

        self.client.patch(
            f"/api/v2/projects/{project_id}/generation-mode",
            json={"generation_mode": "directed"},
        )
        rollback_resp = self.client.post(
            f"/api/v2/projects/{project_id}/snapshots/{snapshot_id}/rollback"
        )
        self.assertEqual(rollback_resp.status_code, 200)
        rolled_mode = rollback_resp.get_json()["manifest_preview"]["generation_mode"]
        self.assertEqual(rolled_mode, "auto")

        export_resp = self.client.post(f"/api/v2/projects/{project_id}/export")
        self.assertEqual(export_resp.status_code, 201)
        bundle = export_resp.get_json()["bundle"]
        self.assertTrue(bundle["path"].endswith(".zip"))
        file_resp = self.client.get(bundle["url"])
        self.assertEqual(file_resp.status_code, 200)
        file_resp.close()

    def test_natural_language_scene_and_script_inputs(self):
        create_resp = self.client.post(
            "/api/v2/projects",
            json={
                "brief": {
                    "video_type": "story",
                    "goal": "drive installs",
                    "target_audience": "creators",
                    "platform": "TikTok",
                    "duration_seconds": 24,
                    "aspect_ratio": "9:16",
                    "tone": "funny",
                    "language": "English",
                    "accent": "neutral",
                }
            },
        )
        project_id = create_resp.get_json()["project_id"]

        scenes_resp = self.client.patch(
            f"/api/v2/projects/{project_id}/steps/scenes",
            json={
                "scene_plan_text": "Open with a surprising hook\nShow the product in action\nClose with a clear CTA"
            },
        )
        self.assertEqual(scenes_resp.status_code, 200)
        scenes = scenes_resp.get_json()["manifest_preview"]["scenes"]
        self.assertEqual(len(scenes), 3)
        self.assertIn("surprising hook", scenes[0]["scene_goal"].lower())

        script_resp = self.client.patch(
            f"/api/v2/projects/{project_id}/steps/script",
            json={
                "script_text": (
                    "scene_001: Hook moment\n"
                    "char_host: Stop scrolling.\n"
                    "char_friend: This is faster than you think."
                )
            },
        )
        self.assertEqual(script_resp.status_code, 200)
        updated_scenes = script_resp.get_json()["manifest_preview"]["scenes"]
        self.assertTrue(len(updated_scenes[0]["dialogue"]) >= 2)

    def test_character_image_upload_endpoint(self):
        create_resp = self.client.post(
            "/api/v2/projects",
            json={
                "brief": {
                    "video_type": "ad",
                    "goal": "sell course",
                    "target_audience": "students",
                    "platform": "Instagram",
                    "duration_seconds": 16,
                    "aspect_ratio": "9:16",
                    "tone": "serious",
                    "language": "English",
                    "accent": "neutral",
                }
            },
        )
        project_id = create_resp.get_json()["project_id"]
        self.client.patch(
            f"/api/v2/projects/{project_id}/steps/characters",
            json={
                "characters": [
                    {
                        "character_id": "char_ava",
                        "name": "Ava",
                        "role": "host",
                        "appearance": "Blue jacket",
                        "personality": "confident",
                        "constraints": "Avoid resemblance to real people.",
                        "voice": {"voice_provider": "native", "voice_id": "native_default"},
                    }
                ]
            },
        )

        upload_resp = self.client.post(
            f"/api/v2/projects/{project_id}/characters/char_ava/images",
            data={"image": (io.BytesIO(b"fakeimagebytes"), "ava.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(upload_resp.status_code, 201)
        payload = upload_resp.get_json()
        self.assertEqual(payload["character_id"], "char_ava")
        self.assertTrue(payload["uploaded_images"])
        bible_refs = payload["manifest_preview"]["character_bible"]["char_ava"]["reference_images"]
        self.assertTrue(len(bible_refs) >= 1)

    def test_character_image_upload_endpoint_accepts_patch(self):
        create_resp = self.client.post(
            "/api/v2/projects",
            json={
                "brief": {
                    "video_type": "ad",
                    "goal": "sell course",
                    "target_audience": "students",
                    "platform": "Instagram",
                    "duration_seconds": 16,
                    "aspect_ratio": "9:16",
                    "tone": "serious",
                    "language": "English",
                    "accent": "neutral",
                }
            },
        )
        project_id = create_resp.get_json()["project_id"]
        self.client.patch(
            f"/api/v2/projects/{project_id}/steps/characters",
            json={
                "characters": [
                    {
                        "character_id": "char_ava",
                        "name": "Ava",
                        "role": "host",
                        "appearance": "Blue jacket",
                        "personality": "confident",
                        "constraints": "Avoid resemblance to real people.",
                        "voice": {"voice_provider": "native", "voice_id": "native_default"},
                    }
                ]
            },
        )

        upload_resp = self.client.patch(
            f"/api/v2/projects/{project_id}/characters/char_ava/images",
            data={"image": (io.BytesIO(b"fakeimagebytes"), "ava.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(upload_resp.status_code, 201)
        payload = upload_resp.get_json()
        self.assertEqual(payload["character_id"], "char_ava")
        self.assertTrue(payload["uploaded_images"])

    def test_description_only_project_bootstraps_manifest(self):
        resp = self.client.post(
            "/api/v2/projects",
            json={
                "description": (
                    "Create a funny TikTok ad for students, 15 seconds, English with neutral accent, "
                    "goal is to drive signups for a study app."
                )
            },
        )
        self.assertEqual(resp.status_code, 201)
        payload = resp.get_json()
        manifest = payload["manifest_preview"]
        completion = payload["completion"]

        self.assertEqual(manifest["brief"]["platform"], "TikTok")
        self.assertTrue(manifest["brief"]["video_type"])
        self.assertTrue(manifest["brief"]["goal"])
        self.assertTrue(manifest["characters"])
        self.assertTrue(manifest["scenes"])
        self.assertTrue(completion["missing_required_brief_fields"] == [])

    def test_description_planner_patch_merges_into_manifest(self):
        class StubPlanner:
            def plan_manifest_patch(self, description, context=None):
                return {
                    "brief": {
                        "video_type": "tutorial",
                        "goal": "Drive demo bookings.",
                        "target_audience": "startup founders",
                        "platform": "YouTube",
                        "duration_seconds": 24,
                        "aspect_ratio": "16:9",
                        "tone": "serious",
                        "language": "English",
                        "accent": "British",
                    },
                    "characters": [
                        {
                            "name": "Mia",
                            "role": "host",
                            "appearance": "Age 28-35, navy blazer, short curly hair",
                            "personality": "clear and calm",
                            "constraints": "Avoid resemblance to real people.",
                        }
                    ],
                    "world": {
                        "setting": "Modern studio, morning",
                        "camera_style": "tripod medium shots",
                    },
                    "scene_plan_text": (
                        "Hook the pain point quickly\n"
                        "Show the workflow in two steps\n"
                        "Close with a direct demo CTA"
                    ),
                    "script_text": (
                        "scene_001: Hook\n"
                        "mia: You can cut your onboarding time in half.\n"
                        "mia: Book a quick demo to see it live."
                    ),
                    "assumptions": [
                        "Defaulted to YouTube because tutorial style was requested.",
                    ],
                }

        services = self.app.config["V2_SERVICES"]
        manifest_service = services.manifest_service
        original_planner = manifest_service.description_planner_service
        manifest_service.description_planner_service = StubPlanner()
        try:
            resp = self.client.post(
                "/api/v2/projects",
                json={
                    "description": "Make a tutorial video that explains our SaaS in a professional style."
                },
            )
        finally:
            manifest_service.description_planner_service = original_planner

        self.assertEqual(resp.status_code, 201)
        payload = resp.get_json()
        manifest = payload["manifest_preview"]
        self.assertEqual(manifest["brief"]["video_type"], "tutorial")
        self.assertEqual(manifest["brief"]["platform"], "YouTube")
        self.assertEqual(manifest["brief"]["aspect_ratio"], "16:9")
        self.assertTrue(manifest["characters"])
        self.assertEqual(manifest["characters"][0]["name"], "Mia")
        self.assertTrue(manifest["scenes"])
        self.assertTrue(manifest.get("planner_assumptions"))
        self.assertTrue(
            any(
                item.get("source") == "gemini_inference"
                for item in manifest.get("autofill_log", [])
            )
        )

    def test_scene_type_can_be_selected_on_scenes_step(self):
        create_resp = self.client.post(
            "/api/v2/projects",
            json={
                "brief": {
                    "video_type": "ad",
                    "goal": "drive installs",
                    "target_audience": "students",
                    "platform": "TikTok",
                    "duration_seconds": 24,
                    "aspect_ratio": "9:16",
                    "tone": "funny",
                    "language": "English",
                    "accent": "neutral",
                }
            },
        )
        self.assertEqual(create_resp.status_code, 201)
        project_id = create_resp.get_json()["project_id"]

        scenes_resp = self.client.patch(
            f"/api/v2/projects/{project_id}/steps/scenes",
            json={
                "scene_plan_text": "Fast opener\nShow social proof\nClose CTA",
                "scene_type_default": "hook",
            },
        )
        self.assertEqual(scenes_resp.status_code, 200)
        scenes = scenes_resp.get_json()["manifest_preview"]["scenes"]
        self.assertEqual(len(scenes), 3)
        self.assertTrue(all(scene.get("scene_type") == "hook" for scene in scenes))

        manual_resp = self.client.patch(
            f"/api/v2/projects/{project_id}/steps/scenes",
            json={
                "scenes": [
                    {
                        "scene_id": "scene_001",
                        "scene_goal": "Hook fast with relatable opener",
                        "scene_type": "hook",
                    },
                    {
                        "scene_id": "scene_002",
                        "scene_goal": "Validate pain and show solution",
                        "scene_type": "solution",
                    },
                    {
                        "scene_id": "scene_003",
                        "scene_goal": "Clear action close",
                        "scene_type": "cta",
                    },
                ]
            },
        )
        self.assertEqual(manual_resp.status_code, 200)
        manual_scenes = manual_resp.get_json()["manifest_preview"]["scenes"]
        self.assertEqual(
            [scene.get("scene_type") for scene in manual_scenes],
            ["hook", "solution", "cta"],
        )

    def test_scene_count_hint_controls_generated_scene_count(self):
        create_resp = self.client.post(
            "/api/v2/projects",
            json={
                "brief": {
                    "video_type": "ad",
                    "goal": "drive installs",
                    "target_audience": "students",
                    "platform": "TikTok",
                    "duration_seconds": 24,
                    "scene_count_hint": 4,
                    "aspect_ratio": "9:16",
                    "tone": "funny",
                    "language": "English",
                    "accent": "neutral",
                }
            },
        )
        self.assertEqual(create_resp.status_code, 201)
        project_id = create_resp.get_json()["project_id"]

        scenes_resp = self.client.post(f"/api/v2/projects/{project_id}/steps/scenes/skip")
        self.assertEqual(scenes_resp.status_code, 200)
        scenes = scenes_resp.get_json()["manifest_preview"]["scenes"]
        self.assertEqual(len(scenes), 4)

    def test_description_scene_headers_preserve_scene_count_and_name(self):
        resp = self.client.post(
            "/api/v2/projects",
            json={
                "description": (
                    "Character Profile: Sue\n"
                    "Name: Sue\n"
                    "Scene 1: Hook\n"
                    "Scene 2: Problem\n"
                    "Scene 3: Reframe\n"
                    "Scene 4: Solution\n"
                    "Scene 5: Payoff\n"
                    "Scene 6: CTA\n"
                    "30 seconds vertical video for TikTok."
                )
            },
        )
        self.assertEqual(resp.status_code, 201)
        payload = resp.get_json()
        manifest = payload["manifest_preview"]
        self.assertEqual(payload["completion"]["missing_required_brief_fields"], [])
        self.assertGreaterEqual(len(manifest.get("scenes", [])), 6)
        self.assertEqual(manifest.get("characters", [])[0].get("name"), "Sue")
        first_scene = manifest.get("scenes", [])[0]
        self.assertIn(first_scene.get("scene_type"), {"hook", "auto"})

    def test_simple_video_endpoint_requires_prompt(self):
        response = self.client.post("/api/simple-video", data={})
        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertEqual(payload["error"], "Prompt is required.")

    def test_simple_video_endpoint_accepts_prompt_and_optional_image(self):
        simple_video_service = self.app.config["SIMPLE_VIDEO_SERVICE"]
        with patch.object(
            simple_video_service,
            "generate_video",
            return_value=SimpleVideoResult(
                output_filename="single_video_test123.mp4",
                output_path=os.path.join(self.temp_dir, "output", "single_video_test123.mp4"),
                provider="gemini",
                model_name="veo-3.1-generate-preview",
                used_reference_image=True,
                aspect_ratio="16:9",
            ),
        ) as mocked_generate:
            response = self.client.post(
                "/api/simple-video",
                data={
                    "prompt": "A clean product shot with subtle camera motion.",
                    "image": (io.BytesIO(b"fakeimagebytes"), "reference.jpg"),
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["filename"], "single_video_test123.mp4")
        self.assertEqual(payload["videoUrl"], "/output/single_video_test123.mp4")
        self.assertEqual(payload["provider"], "gemini")
        self.assertEqual(payload["model"], "veo-3.1-generate-preview")
        self.assertTrue(payload["hasReferenceImage"])
        self.assertEqual(payload["aspectRatio"], "16:9")
        mocked_generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
