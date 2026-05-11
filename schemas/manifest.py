from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


REQUIRED_BRIEF_FIELDS = [
    "video_type",
    "goal",
    "target_audience",
    "platform",
    "duration_seconds",
    "aspect_ratio",
    "tone",
    "language",
    "accent",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AutoFillDecision:
    field_path: str
    value: Any
    source: str = "autofill"
    reason: str = ""
    generator_version: str = "v2.0"
    seed_fragment: str = ""
    timestamp: str = field(default_factory=utc_now_iso)


@dataclass
class SafetyWarning:
    category: str
    severity: str
    message: str
    field_path: str
    timestamp: str = field(default_factory=utc_now_iso)


@dataclass
class CharacterVoice:
    voice_provider: str = "native"
    voice_id: str = "native_default"
    speaking_style: str = "clear, natural, medium pace"


@dataclass
class CharacterSpec:
    character_id: str
    name: str
    role: str
    appearance: str
    personality: str
    constraints: str = "Avoid resemblance to real people."
    voice: CharacterVoice = field(default_factory=CharacterVoice)


@dataclass
class DialogueLine:
    speaker_id: str
    line: str
    emotion: str
    pause_notes: str = ""
    start_second: float = 0.0
    end_second: float = 0.0


@dataclass
class ShotSpec:
    shot_type: str
    camera_move: str
    subject_focus: str
    duration_seconds: float
    on_screen_text: str = ""
    music_mood: str = ""
    sound_effects: List[str] = field(default_factory=list)
    transition_out: str = "cut"


@dataclass
class SceneSpec:
    scene_id: str
    scene_goal: str
    location_time: str
    scene_type: str = "auto"
    characters_in_scene: List[str] = field(default_factory=list)
    events: List[str] = field(default_factory=list)
    dialogue: List[DialogueLine] = field(default_factory=list)
    narration: str = ""
    shot_list: List[ShotSpec] = field(default_factory=list)
    locked: bool = False
    state: str = "planned"
    state_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class VideoBrief:
    description: str = ""
    video_type: Optional[str] = None
    goal: Optional[str] = None
    target_audience: Optional[str] = None
    platform: Optional[str] = None
    duration_seconds: Optional[int] = None
    aspect_ratio: Optional[str] = None
    tone: Optional[str] = None
    language: Optional[str] = None
    accent: Optional[str] = None
    brand_rules: str = ""
    logline: str = ""


@dataclass
class WorldStyle:
    setting: str = ""
    background_style: str = ""
    visual_references: List[str] = field(default_factory=list)
    color_palette: str = ""
    wardrobe_props: str = ""
    camera_style: str = ""
    lighting: str = ""
    motion: str = ""


@dataclass
class OutputSpec:
    resolution: str = "1080p"
    fps: int = 30
    subtitles_enabled: bool = True
    subtitle_style: str = "clean lower-third"
    safe_areas: str = "social-safe margins"
    file_format: str = "mp4"
    deliverables: List[str] = field(
        default_factory=lambda: [
            "final_video",
            "script",
            "scene_list",
            "assets_list",
        ]
    )


@dataclass
class RenderJob:
    job_id: str
    project_id: str
    status: str
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    progress: float = 0.0
    warnings: List[SafetyWarning] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    artifacts: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class VideoManifest:
    project_id: str
    created_at: str
    updated_at: str
    project_seed: str
    generation_mode: str = "auto"
    surprise_counters: Dict[str, int] = field(default_factory=dict)
    brief: VideoBrief = field(default_factory=VideoBrief)
    characters: List[CharacterSpec] = field(default_factory=list)
    world: WorldStyle = field(default_factory=WorldStyle)
    scenes: List[SceneSpec] = field(default_factory=list)
    output: OutputSpec = field(default_factory=OutputSpec)
    audio: Dict[str, Any] = field(
        default_factory=lambda: {
            "music_mood": "",
            "sound_effects": [],
            "narrator_track": None,
        }
    )
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    safety_warnings: List[SafetyWarning] = field(default_factory=list)
    autofill_log: List[AutoFillDecision] = field(default_factory=list)
    scene_variants: Dict[str, Any] = field(default_factory=dict)
    character_bible: Dict[str, Any] = field(default_factory=dict)
    approvals: Dict[str, Any] = field(default_factory=dict)
    render_iteration: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
