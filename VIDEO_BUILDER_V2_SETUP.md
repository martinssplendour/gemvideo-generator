# Video Builder V2 Manual Setup

## Prerequisites
1. Python 3.11+.
2. FFmpeg installed and available on PATH.
3. Gemini API key (`GEMINI_API_KEY`).
4. Optional: ElevenLabs key (`ELEVENLABS_API_KEY`) for cloud TTS.

## Environment variables
Use `.env` (or system env):

```env
GEMINI_API_KEY=your_key
ELEVENLABS_API_KEY=
DATA_DIR=data
OUTPUT_DIR=output
UPLOAD_DIR=uploads
ENABLE_REAL_GENERATION=true
VEO_MODEL_NAME=veo-3.1-generate-preview
```

## Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## Run service

Start Flask app:
```powershell
python application.py
```

## Health check
Open:
`GET /api/v2/health`

Response includes FFmpeg status and local render-runner mode.

## Control endpoints
- `POST /api/v2/projects/{project_id}/approvals/script` to approve/unapprove script.
- `POST /api/v2/jobs/{job_id}/pause` to pause generation.
- `POST /api/v2/jobs/{job_id}/resume` to resume generation.
- `POST /api/v2/jobs/{job_id}/cancel` to cancel generation.
- `POST /api/v2/projects/{project_id}/scenes/{scene_id}/regenerate` for scene-level regenerate/branch.
- `PATCH /api/v2/projects/{project_id}/timeline` for timeline/transitions/B-roll controls.
- `PATCH /api/v2/projects/{project_id}/character-bible` for visual consistency memory.

## Legacy compatibility
- Existing endpoint stays available: `POST /api/generate`.
- New V2 flow is available from `index.html` and `/api/v2/*` endpoints.
