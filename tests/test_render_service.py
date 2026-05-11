import copy
import os
import tempfile
import unittest
from unittest.mock import patch

from services.render_service import RenderService


class _DummyManifestService:
    def __init__(self, manifest):
        self._manifest = manifest

    def get_manifest(self, project_id):
        return copy.deepcopy(self._manifest)

    def save_manifest(self, project_id, manifest):
        self._manifest = copy.deepcopy(manifest)

    def project_paths(self, project_id):
        raise RuntimeError("stop-after-validation")


class _LoopManifestService:
    def __init__(self, manifest, root_dir):
        self._manifest = manifest
        self._root_dir = root_dir
        self._job = {"status": "running", "progress": 0.0}

    def get_manifest(self, project_id):
        return copy.deepcopy(self._manifest)

    def save_manifest(self, project_id, manifest):
        self._manifest = copy.deepcopy(manifest)

    def project_paths(self, project_id):
        video_dir = os.path.join(self._root_dir, "video")
        audio_dir = os.path.join(self._root_dir, "audio")
        subtitle_dir = os.path.join(self._root_dir, "subtitles")
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(subtitle_dir, exist_ok=True)
        return {
            "project_dir": self._root_dir,
            "assets_video_dir": video_dir,
            "assets_audio_dir": audio_dir,
            "assets_subtitles_dir": subtitle_dir,
        }

    def update_job(self, project_id, job_id, **kwargs):
        self._job.update(kwargs)

    def get_job(self, job_id):
        return dict(self._job)

    def set_scene_state(self, project_id, scene_id, state, note="", force=False):
        for scene in self._manifest.get("scenes", []):
            if scene.get("scene_id") == scene_id:
                scene["state"] = state
                return
        raise FileNotFoundError(scene_id)


class _DummyValidationService:
    def normalize_manifest(self, manifest):
        return manifest

    def ensure_render_eligible(self, manifest):
        if not manifest.get("approvals", {}).get("script_approved"):
            raise AssertionError("script approval was lost before validation")
        raise RuntimeError("stop-after-validation")


class _PassingValidationService:
    def normalize_manifest(self, manifest):
        return manifest

    def ensure_render_eligible(self, manifest):
        return None


class _DummyAutoFillService:
    def autofill_all(self, manifest, reason=""):
        # Simulate current autofill behavior that can reset script approval.
        approvals = manifest.setdefault("approvals", {})
        approvals["script_approved"] = False
        approvals["script_needs_user_approval"] = True
        return manifest


class _DummySafetyService:
    def run_checks(self, manifest):
        return []


class _DummyVoiceService:
    def synthesize_dialogue_tracks(self, manifest, audio_dir):
        return []


class _DummyReportService:
    pass


class _DummyVeoClientService:
    pass


class RenderServiceTests(unittest.TestCase):
    def _build_render_service(self, manifest=None):
        return RenderService(
            manifest_service=_DummyManifestService(manifest or {}),
            validation_service=_DummyValidationService(),
            autofill_service=_DummyAutoFillService(),
            safety_service=_DummySafetyService(),
            voice_service=_DummyVoiceService(),
            report_service=_DummyReportService(),
            veo_client_service=_DummyVeoClientService(),
            enable_real_generation=False,
        )

    def _build_loop_render_service(self, manifest, root_dir):
        return RenderService(
            manifest_service=_LoopManifestService(manifest or {}, root_dir),
            validation_service=_PassingValidationService(),
            autofill_service=_DummyAutoFillService(),
            safety_service=_DummySafetyService(),
            voice_service=_DummyVoiceService(),
            report_service=_DummyReportService(),
            veo_client_service=_DummyVeoClientService(),
            enable_real_generation=False,
        )

    def test_preserves_script_approval_after_render_autofill(self):
        manifest = {
            "project_id": "p1",
            "project_seed": "s1",
            "generation_mode": "auto",
            "brief": {
                "video_type": "ad",
                "goal": "drive signups",
                "target_audience": "students",
                "platform": "TikTok",
                "duration_seconds": 15,
                "aspect_ratio": "9:16",
                "tone": "funny",
                "language": "English",
                "accent": "neutral",
            },
            "scenes": [
                {
                    "scene_id": "scene_001",
                    "scene_goal": "Hook",
                    "scene_type": "hook",
                    "dialogue": [{"speaker_id": "char_1", "line": "Hi"}],
                    "shot_list": [{"duration_seconds": 2.0}],
                }
            ],
            "output": {"subtitles_enabled": False},
            "audio": {},
            "approvals": {
                "script_approved": True,
                "script_needs_user_approval": False,
                "script_last_reviewed_at": "2026-02-21T00:00:00+00:00",
            },
        }
        render_service = self._build_render_service(manifest)

        with self.assertRaisesRegex(RuntimeError, "stop-after-validation"):
            render_service.render_project("p1", "j1")

    def test_select_reference_inputs_for_first_and_later_scene(self):
        render_service = self._build_render_service({})
        reference_inputs = [
            {"path": "C:/tmp/char.jpg", "role": "character"},
            {"path": "C:/tmp/world.jpg", "role": "world_style"},
            {"path": "C:/tmp/frame.jpg", "role": "continuity"},
        ]
        first_scene = render_service._select_reference_inputs_for_generation(
            reference_inputs=reference_inputs,
            scene_index=0,
        )
        later_scene = render_service._select_reference_inputs_for_generation(
            reference_inputs=reference_inputs,
            scene_index=3,
        )
        self.assertEqual(len(first_scene), 1)
        self.assertEqual(first_scene[0]["role"], "character")
        self.assertEqual(len(later_scene), 1)
        self.assertEqual(later_scene[0]["role"], "continuity")

    def test_select_reference_inputs_without_continuity_uses_opening_style(self):
        render_service = self._build_render_service({})
        reference_inputs = [
            {"path": "C:/tmp/char.jpg", "role": "character"},
            {"path": "C:/tmp/frame.jpg", "role": "continuity"},
        ]
        later_scene = render_service._select_reference_inputs_for_generation(
            reference_inputs=reference_inputs,
            scene_index=2,
            continuity_mode=False,
        )
        self.assertEqual(len(later_scene), 1)
        self.assertEqual(later_scene[0]["role"], "character")

    def test_should_use_continuity_anchor_skips_when_cast_changes(self):
        render_service = self._build_render_service({})
        scenes = [
            {"scene_id": "scene_001", "characters_in_scene": ["char_sarah"]},
            {
                "scene_id": "scene_002",
                "characters_in_scene": ["char_mark"],
                "continuity": {"must_follow_previous_frame": True},
            },
        ]
        use_continuity, reason = render_service._should_use_continuity_anchor(
            scene_index=1,
            scene=scenes[1],
            scenes=scenes,
        )
        self.assertFalse(use_continuity)
        self.assertEqual(reason, "cast_mismatch")

    def test_collect_reference_inputs_ignores_scene_start_frame_reference(self):
        render_service = self._build_render_service({})
        with tempfile.TemporaryDirectory() as tmp_dir:
            char_ref_path = os.path.join(tmp_dir, "char_ref.png")
            start_ref_path = os.path.join(tmp_dir, "start_ref.png")
            with open(char_ref_path, "wb") as file:
                file.write(b"char")
            with open(start_ref_path, "wb") as file:
                file.write(b"start")

            manifest = {
                "character_bible": {
                    "char_1": {
                        "reference_images": ["char_ref.png"],
                    }
                },
                "world": {},
            }
            scene = {
                "characters_in_scene": ["char_1"],
                "reference_images": [],
                "continuity": {
                    "start_frame_reference": "start_ref.png",
                },
            }
            resolved = render_service._collect_reference_image_paths(
                manifest=manifest,
                scene=scene,
                continuity_frame_path=None,
                project_dir=tmp_dir,
            )
            roles = {item.get("role") for item in resolved}
            self.assertIn("character", roles)
            self.assertNotIn("continuity", roles)

    def test_collect_reference_inputs_accepts_repo_relative_continuity_path(self):
        render_service = self._build_render_service({})
        repo_root = os.getcwd()
        with tempfile.TemporaryDirectory(dir=repo_root) as tmp_dir:
            project_dir = os.path.join(tmp_dir, "data", "projects", "p1")
            frame_dir = os.path.join(project_dir, "assets", "video", "frames")
            os.makedirs(frame_dir, exist_ok=True)
            frame_path = os.path.join(frame_dir, "end.jpg")
            with open(frame_path, "wb") as file:
                file.write(b"frame")

            relative_project_dir = os.path.relpath(project_dir, repo_root)
            relative_frame_path = os.path.relpath(frame_path, repo_root)
            resolved = render_service._collect_reference_image_paths(
                manifest={"character_bible": {}, "world": {}},
                scene={"characters_in_scene": [], "reference_images": []},
                continuity_frame_path=relative_frame_path,
                project_dir=relative_project_dir,
            )

            self.assertEqual(len(resolved), 1)
            self.assertEqual(resolved[0]["role"], "continuity")
            self.assertTrue(
                resolved[0]["path"].endswith(os.path.join("frames", "end.jpg"))
            )

    def test_build_reference_images_returns_typed_reference_payload(self):
        render_service = self._build_render_service({})
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = os.path.join(tmp_dir, "ref.png")
            with open(image_path, "wb") as file:
                file.write(b"fake-image-bytes")

            refs = render_service._build_reference_images(
                [{"path": image_path, "role": "character"}]
            )

            self.assertEqual(len(refs), 1)
            self.assertTrue(hasattr(refs[0], "image"))
            image = getattr(refs[0], "image", None)
            self.assertIsNotNone(image)
            self.assertTrue(getattr(image, "image_bytes", b""))

    def test_ensure_continuity_input_available_for_later_scene(self):
        render_service = self._build_render_service({})
        with self.assertRaisesRegex(RuntimeError, "missing previous scene end frame"):
            render_service._ensure_continuity_input_available(
                scene_index=2,
                scene_id="scene_003",
                scene={"continuity": {"must_follow_previous_frame": True}},
                continuity_frame_path=None,
            )

    def test_ensure_continuity_input_available_skips_when_not_required(self):
        render_service = self._build_render_service({})
        render_service._ensure_continuity_input_available(
            scene_index=2,
            scene_id="scene_003",
            scene={"continuity": {"must_follow_previous_frame": True}},
            continuity_frame_path=None,
            require_continuity=False,
        )

    def test_ensure_last_frame_anchor_ready_for_later_scene(self):
        render_service = self._build_render_service({})
        with self.assertRaisesRegex(RuntimeError, "could not be loaded as a last-frame anchor"):
            render_service._ensure_last_frame_anchor_ready(
                scene_index=2,
                scene_id="scene_003",
                scene={"continuity": {"must_follow_previous_frame": True}},
                continuity_frame_path="C:/tmp/frame.jpg",
                last_frame_image=None,
            )

    def test_ensure_last_frame_anchor_ready_skips_when_not_required(self):
        render_service = self._build_render_service({})
        render_service._ensure_last_frame_anchor_ready(
            scene_index=2,
            scene_id="scene_003",
            scene={"continuity": {"must_follow_previous_frame": True}},
            continuity_frame_path=None,
            last_frame_image=None,
            require_continuity=False,
        )

    def test_seed_continuity_from_previous_scene_prefers_saved_end_frame(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_dir = os.path.join(tmp_dir, "video")
            frames_dir = os.path.join(video_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)

            saved_end_frame = os.path.join(frames_dir, "proj1_scene_001_001_end.jpg")
            with open(saved_end_frame, "wb") as file:
                file.write(b"frame")

            manifest = {
                "scenes": [
                    {
                        "scene_id": "scene_001",
                        "render_debug": {
                            "continuity_frame_path": "assets/video/frames/proj1_scene_001_001_end.jpg",
                            "output_clip_path": "assets/video/proj1_scene_001_001.mp4",
                        },
                    },
                    {
                        "scene_id": "scene_002",
                    },
                ]
            }
            render_service = self._build_render_service(manifest)

            resolved = render_service._seed_continuity_from_previous_scene(
                manifest=manifest,
                project_id="proj1",
                target_scene_id="scene_002",
                project_dir=tmp_dir,
                video_dir=video_dir,
                frames_dir=frames_dir,
                warnings=[],
            )

            self.assertEqual(resolved, saved_end_frame)

    def test_render_does_not_rewait_for_script_approval_inside_scene_loop(self):
        manifest = {
            "project_id": "p1",
            "project_seed": "s1",
            "generation_mode": "auto",
            "brief": {
                "video_type": "ad",
                "goal": "drive signups",
                "target_audience": "students",
                "platform": "TikTok",
                "duration_seconds": 15,
                "aspect_ratio": "9:16",
                "tone": "funny",
                "language": "English",
                "accent": "neutral",
            },
            "characters": [],
            "world": {},
            "scenes": [
                {
                    "scene_id": "scene_001",
                    "scene_goal": "Hook",
                    "scene_type": "hook",
                    "dialogue": [],
                    "shot_list": [],
                    "continuity": {"must_follow_previous_frame": False},
                }
            ],
            "output": {"subtitles_enabled": False},
            "audio": {},
            "approvals": {
                "script_approved": True,
                "script_needs_user_approval": False,
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            render_service = self._build_loop_render_service(manifest, tmp_dir)
            with patch.object(
                render_service,
                "_wait_for_script_approval",
                side_effect=AssertionError("should not wait"),
            ):
                with patch.object(
                    render_service,
                    "_ensure_continuity_input_available",
                    side_effect=RuntimeError("entered-scene-loop"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "entered-scene-loop"):
                        render_service.render_project("p1", "j1")


if __name__ == "__main__":
    unittest.main()
