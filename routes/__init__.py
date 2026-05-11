from routes.v2_jobs import create_v2_jobs_blueprint
from routes.v2_projects import create_v2_projects_blueprint
from routes.v2_voices import create_v2_voices_blueprint

__all__ = [
    "create_v2_jobs_blueprint",
    "create_v2_projects_blueprint",
    "create_v2_voices_blueprint",
]
