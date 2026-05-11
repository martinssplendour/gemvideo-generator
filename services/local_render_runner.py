from __future__ import annotations

import threading
from typing import Dict, Optional

from services.manifest_service import ManifestService
from services.render_service import RenderCancelled, RenderService


class LocalRenderRunner:
    """Runs render jobs in background threads for single-instance local use."""

    def __init__(
        self,
        manifest_service: ManifestService,
        render_service: RenderService,
    ):
        self.manifest_service = manifest_service
        self.render_service = render_service
        self._lock = threading.Lock()
        self._active_threads: Dict[str, threading.Thread] = {}
        self._project_active_job: Dict[str, str] = {}

    def start(self, project_id: str, job_id: str) -> None:
        self.start_with_options(project_id, job_id)

    def start_with_options(
        self,
        project_id: str,
        job_id: str,
        *,
        target_scene_id: Optional[str] = None,
        scene_only: bool = False,
    ) -> None:
        with self._lock:
            existing_job = self._project_active_job.get(project_id)
            if existing_job:
                thread = self._active_threads.get(existing_job)
                if thread and thread.is_alive():
                    raise ValueError(
                        f"Project '{project_id}' already has an active render job '{existing_job}'."
                    )
                self._project_active_job.pop(project_id, None)
                self._active_threads.pop(existing_job, None)

            worker = threading.Thread(
                target=self._run_job,
                args=(project_id, job_id, target_scene_id, scene_only),
                daemon=True,
                name=f"render-{project_id}-{job_id}",
            )
            self._active_threads[job_id] = worker
            self._project_active_job[project_id] = job_id
            worker.start()

    def is_running(self, job_id: str) -> bool:
        with self._lock:
            thread = self._active_threads.get(job_id)
            return bool(thread and thread.is_alive())

    def _run_job(
        self,
        project_id: str,
        job_id: str,
        target_scene_id: Optional[str] = None,
        scene_only: bool = False,
    ) -> None:
        def progress(stage: str, value: float, note: Optional[str]) -> None:
            self.manifest_service.update_job(
                project_id,
                job_id,
                status="running",
                progress=value,
                current_stage=stage,
                current_scene_id=note,
            )

        try:
            self.manifest_service.update_job(
                project_id,
                job_id,
                status="running",
                progress=0.02,
                current_stage="starting",
            )
            result = self.render_service.render_project(
                project_id=project_id,
                job_id=job_id,
                progress_cb=progress,
                target_scene_id=target_scene_id,
                scene_only=scene_only,
            )
            self.manifest_service.update_job(
                project_id,
                job_id,
                status="completed",
                progress=1.0,
                warnings=result.get("warnings", []),
                artifacts=result.get("artifacts", []),
            )
        except RenderCancelled as error:
            self.manifest_service.update_job(
                project_id,
                job_id,
                status="cancelled",
                append_error=str(error),
            )
        except Exception as error:
            self.manifest_service.update_job(
                project_id,
                job_id,
                status="failed",
                append_error=str(error),
            )
        finally:
            with self._lock:
                self._active_threads.pop(job_id, None)
                active_job = self._project_active_job.get(project_id)
                if active_job == job_id:
                    self._project_active_job.pop(project_id, None)
