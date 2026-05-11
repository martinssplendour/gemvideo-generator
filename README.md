# GemVideo Generator

GemVideo Generator is a Flask-based AI video generation prototype for turning user prompts, scripts, and scene plans into structured video generation jobs. It is designed as a portfolio-ready backend and web interface for experimenting with prompt-to-video workflows, provider configuration, scene planning, and local render orchestration.

## What This Project Does

The app accepts high-level creative input and breaks it into a more controlled video-production workflow. Instead of treating video generation as a single prompt, it models projects, jobs, scenes, manifests, safety checks, provider calls, and render state. That structure makes it easier to regenerate individual scenes, adjust timeline details, keep character descriptions consistent, and test generation providers without rewriting the whole app.

## The Work Behind The Project

This project was built around the idea that AI video generation needs orchestration, not just a prompt box. A useful video workflow has to keep track of creative direction, scene breakdowns, provider settings, safety checks, retries, assets, outputs, and job state. The implementation separates those responsibilities into routes, schemas, and services so the workflow can be tested and extended piece by piece.

The core work included designing a second-generation job system with explicit project and job APIs. A project can hold the creative context, while a job can represent a specific generation attempt with scenes, prompts, render state, and output metadata. This makes the app closer to a lightweight production pipeline than a one-off Flask demo.

The service layer was split into focused components:

- `description_planner_service.py` and `scene_builder_service.py` handle prompt expansion and scene planning.
- `manifest_service.py` and `schemas/manifest.py` define the structured state of a generation job.
- `safety_service.py` and `validation_service.py` keep validation and safety checks out of route handlers.
- `render_service.py`, `local_render_runner.py`, and provider client services isolate rendering from orchestration.
- `voice_service.py` supports voice/TTS-style extensions without coupling them to the main app.

This structure lets the app run in a local simulation mode while still leaving clear integration points for real Gemini/Veo, OpenAI-style video, or voice providers.

## Workflow

1. A user creates or opens a video project.
2. The app turns a brief prompt or script into structured scene descriptions.
3. The job manifest records scenes, timing, provider settings, and render status.
4. Validation and safety services check the request before generation.
5. The render service either calls a configured provider or uses local render-runner mode.
6. Individual scenes can be regenerated without rebuilding the entire project.
7. Output metadata is stored separately from private/generated media.

## Key Features

- Flask web application with a local browser interface.
- V2 project and job APIs for managing video generation work.
- Scene planning services for prompt expansion and structured descriptions.
- Scene-level regenerate, pause, resume, and cancel controls.
- Manifest schema for tracking jobs, scenes, assets, and render state.
- Character-bible style consistency support across scenes.
- Safety and validation services before generation.
- Optional provider clients for Gemini/Veo, OpenAI-style video APIs, and voice generation.
- Local render-runner mode for development without requiring real provider calls.
- Unit tests covering planning, rendering, safety, and V2 API behaviour.

## Tech Stack

- Python
- Flask
- Pydantic-style schema organisation
- Provider service layer for AI/video integrations
- Pytest-compatible tests

## What This Demonstrates

- Backend decomposition for AI workflow orchestration.
- API design for long-running creative jobs.
- State modelling with manifests instead of loose runtime variables.
- Provider abstraction so external AI services can be swapped or mocked.
- Testing around planning, scene building, safety, rendering, and API behaviour.
- Practical cleanup for public release by excluding generated media and secrets.

## Repository Structure

```text
app.py                         # Flask application entry point
application.py                 # Alternate runtime entry point
config.py                      # Environment-driven app configuration
index.html                     # Local web interface
routes/                        # API route modules
schemas/                       # Manifest and request/response structures
services/                      # Planning, rendering, safety, provider, and storage logic
tests/                         # Automated tests
VIDEO_BUILDER_V2_SETUP.md      # Additional setup notes for the V2 workflow
requirements.txt               # Python dependencies
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

On macOS/Linux, activate the environment with:

```bash
source .venv/bin/activate
```

Then edit `.env` with your own API keys if you want to test real provider integrations. Leave `ENABLE_REAL_GENERATION=false` for local development without external generation calls.

## Run Locally

```bash
python application.py
```

Open the local URL printed by Flask in your browser.

## Test

```bash
pytest
```

The tests focus on the service and API layers that are most important to the workflow: planning, safety checks, scene construction, rendering behaviour, and V2 endpoints.

## Environment Variables

The repo includes `.env.example` only. Real keys should stay local and must not be committed.

Common variables:

- `GEMINI_API_KEY`
- `OPENAI_API_KEY`
- `ELEVENLABS_API_KEY`
- `DATA_DIR`
- `OUTPUT_DIR`
- `UPLOAD_DIR`
- `ENABLE_REAL_GENERATION`
- `VEO_MODEL_NAME`

## Public Repository Notes

This cleaned version intentionally excludes local `.env` files, uploads, generated videos, provider outputs, temp folders, caches, and virtual environments. The repository is intended to show application structure, service design, and AI workflow orchestration without publishing private keys or generated media.
