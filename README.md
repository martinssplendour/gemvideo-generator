# GemVideo Generator

GemVideo Generator is a Flask-based AI video generation prototype for turning prompts, scripts, and scene plans into structured video-generation jobs.

## Overview

The project provides a local web interface and API routes for prompt-driven video workflows. It supports project/job control, scene regeneration, timeline edits, character-bible style consistency, and provider configuration for Gemini/OpenAI-style video generation services.

## Features

- Prompt-to-video generation workflow.
- V2 project and job APIs.
- Scene-level regenerate, pause, resume, and cancel controls.
- Timeline and transition editing endpoints.
- Character-bible memory for visual consistency.
- Local render-runner mode for development.
- Optional cloud TTS/video provider configuration.

## Project Structure

```text
app.py
application.py
config.py
index.html
routes/
schemas/
services/
tests/
VIDEO_BUILDER_V2_SETUP.md
requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Create a `.env` file from `.env.example` and add your own API keys.

## Run

```bash
python application.py
```

## Public Repo Notes

Uploads, generated videos, output files, temp folders, local `.env` files, and cache folders are intentionally excluded from this repository.

