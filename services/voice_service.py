from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import urllib.error
import urllib.request
import wave
from array import array
from pathlib import Path
from typing import Any, Dict, List, Optional


class VoiceService:
    def __init__(
        self,
        elevenlabs_api_key: str = "",
        elevenlabs_model_id: str = "eleven_multilingual_v2",
        native_tts_engine: str = "pyttsx3",
    ):
        self.elevenlabs_api_key = elevenlabs_api_key
        self.elevenlabs_model_id = elevenlabs_model_id
        self.native_tts_engine = native_tts_engine

    def list_voices(self) -> Dict[str, List[Dict[str, str]]]:
        elevenlabs_voices = self._list_elevenlabs_voices()
        native_voices = self._list_native_voices()
        return {
            "elevenlabs": elevenlabs_voices,
            "native": native_voices,
        }

    def synthesize_dialogue_tracks(
        self,
        manifest: Dict[str, Any],
        audio_dir: str,
    ) -> List[Dict[str, Any]]:
        os.makedirs(audio_dir, exist_ok=True)
        characters = {
            character.get("character_id"): character
            for character in manifest.get("characters", [])
            if character.get("character_id")
        }

        tracks: List[Dict[str, Any]] = []
        line_index = 0

        for scene in manifest.get("scenes", []):
            scene_id = scene.get("scene_id", "scene_unknown")
            for line in scene.get("dialogue", []):
                line_index += 1
                speaker_id = line.get("speaker_id", "unknown")
                character = characters.get(speaker_id, {})
                voice = character.get("voice", {})

                provider = voice.get("voice_provider", "native")
                voice_id = voice.get("voice_id", f"native_{speaker_id}")
                text = line.get("line", "").strip()
                if not text:
                    continue

                filename = f"{scene_id}_{line_index:03d}_{speaker_id}.wav"
                output_path = Path(audio_dir) / filename

                used_provider = provider
                fallback_reason = ""
                fallback_trace: List[str] = []
                track_path = str(output_path)
                if provider == "elevenlabs":
                    mp3_path = self._try_elevenlabs_tts(text, voice_id, str(output_path))
                    if not mp3_path:
                        used_provider = "native"
                        fallback_reason = "ElevenLabs unavailable or request failed."
                        fallback_trace.append("elevenlabs_failed")
                        native_engine = self._native_spoken_tts(
                            text, voice_id, str(output_path)
                        )
                        fallback_trace.append(native_engine)
                    else:
                        track_path = mp3_path
                        fallback_trace.append("elevenlabs")
                else:
                    native_engine = self._native_spoken_tts(text, voice_id, str(output_path))
                    fallback_trace.append(native_engine)

                start_second = float(line.get("start_second", 0.0))
                end_second = float(line.get("end_second", 0.0))
                qa = self._voice_line_qa(
                    text=text,
                    audio_path=track_path,
                    expected_start=start_second,
                    expected_end=end_second,
                    fallback_reason=fallback_reason,
                    fallback_trace=fallback_trace,
                    requested_provider=provider,
                    used_provider=used_provider,
                )

                tracks.append(
                    {
                        "scene_id": scene_id,
                        "speaker_id": speaker_id,
                        "voice_provider": used_provider,
                        "requested_voice_provider": provider,
                        "voice_id": voice_id,
                        "line": text,
                        "start_second": start_second,
                        "end_second": end_second,
                        "path": track_path.replace("\\", "/"),
                        "fallback_reason": fallback_reason,
                        "fallback_trace": fallback_trace,
                        "qa": qa,
                    }
                )

        return tracks

    def assign_default_voice(
        self,
        project_seed: str,
        character_id: str,
        preferred_provider: str = "native",
    ) -> Dict[str, str]:
        stable = hashlib.sha256(f"{project_seed}:{character_id}".encode("utf-8")).hexdigest()
        if preferred_provider == "elevenlabs" and self.elevenlabs_api_key:
            voices = self._list_elevenlabs_voices()
            if voices:
                index = int(stable[:8], 16) % len(voices)
                selected = voices[index]
                return {
                    "voice_provider": "elevenlabs",
                    "voice_id": selected["voice_id"],
                    "speaking_style": "clear, expressive, medium pace",
                }

        options = ["native_default", "native_warm", "native_bright"]
        selected_id = options[int(stable[:8], 16) % len(options)]
        return {
            "voice_provider": "native",
            "voice_id": selected_id,
            "speaking_style": "clear, expressive, medium pace",
        }

    def _list_elevenlabs_voices(self) -> List[Dict[str, str]]:
        if not self.elevenlabs_api_key:
            return []
        request = urllib.request.Request(
            "https://api.elevenlabs.io/v1/voices",
            headers={"xi-api-key": self.elevenlabs_api_key},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                payload = json.loads(response.read().decode("utf-8"))
                voices = payload.get("voices", [])
                result = []
                for voice in voices:
                    result.append(
                        {
                            "voice_provider": "elevenlabs",
                            "voice_id": voice.get("voice_id", ""),
                            "name": voice.get("name", "Unnamed voice"),
                        }
                    )
                return result
        except Exception:
            return []

    def _list_native_voices(self) -> List[Dict[str, str]]:
        default = [
            {
                "voice_provider": "native",
                "voice_id": "native_default",
                "name": "Native Default",
            }
        ]
        try:
            import pyttsx3  # type: ignore

            engine = pyttsx3.init()
            voices = engine.getProperty("voices") or []
            result = []
            for voice in voices:
                voice_id = str(getattr(voice, "id", "")).strip()
                name = str(getattr(voice, "name", voice_id or "Native Voice")).strip()
                if not voice_id:
                    continue
                result.append(
                    {
                        "voice_provider": "native",
                        "voice_id": voice_id,
                        "name": name,
                    }
                )
            return result or default
        except Exception:
            return default

    def _try_elevenlabs_tts(self, text: str, voice_id: str, output_path: str) -> Optional[str]:
        if not self.elevenlabs_api_key or not voice_id:
            return None
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        payload = json.dumps(
            {
                "text": text,
                "model_id": self.elevenlabs_model_id,
                "voice_settings": {
                    "stability": 0.45,
                    "similarity_boost": 0.7,
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={
                "xi-api-key": self.elevenlabs_api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                audio_bytes = response.read()
            mp3_path = output_path.replace(".wav", ".mp3")
            with open(mp3_path, "wb") as file:
                file.write(audio_bytes)
            return mp3_path
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            return None
        except Exception:
            return None

    def _native_spoken_tts(self, text: str, voice_id: str, output_path: str) -> str:
        if self.native_tts_engine == "pyttsx3":
            try:
                self._native_pyttsx3_tts(text, voice_id, output_path)
                return "pyttsx3"
            except Exception:
                pass
        if os.name == "nt":
            try:
                self._native_windows_sapi_tts(text, output_path)
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    return "windows_sapi"
            except Exception:
                pass
        self._native_tone_tts(text, voice_id, output_path)
        return "synthetic_tone"

    def _native_windows_sapi_tts(self, text: str, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        escaped_text = text.replace("'", "''")
        escaped_output = output_path.replace("'", "''")
        command = (
            "$ErrorActionPreference='Stop';"
            "Add-Type -AssemblyName System.Speech;"
            "$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            f"$voice.SetOutputToWaveFile('{escaped_output}');"
            f"$voice.Speak('{escaped_text}');"
            "$voice.Dispose();"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            check=True,
            capture_output=True,
            text=True,
        )

    def _native_pyttsx3_tts(self, text: str, voice_id: str, output_path: str) -> None:
        import pyttsx3  # type: ignore

        engine = pyttsx3.init()
        voices = engine.getProperty("voices") or []
        selected_voice = None
        voice_lookup = {str(getattr(v, "id", "")): v for v in voices}

        if voice_id in voice_lookup:
            selected_voice = voice_id
        elif voices:
            # Deterministic fallback selection if provided voice ID is unknown.
            index = int(hashlib.sha256(voice_id.encode("utf-8")).hexdigest()[:8], 16) % len(voices)
            selected_voice = str(getattr(voices[index], "id", ""))

        if selected_voice:
            engine.setProperty("voice", selected_voice)

        words = max(1, len(text.split()))
        rate = max(130, min(200, int(180 - (words / 8))))
        engine.setProperty("rate", rate)
        engine.save_to_file(text, output_path)
        engine.runAndWait()

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("pyttsx3 did not produce audio output.")

    def _native_tone_tts(self, text: str, voice_id: str, output_path: str) -> None:
        # Last-resort fallback when native speech synthesis is unavailable.
        sample_rate = 22050
        duration_seconds = max(1.2, min(7.0, len(text.split()) * 0.42))
        samples = int(sample_rate * duration_seconds)
        seed = int(hashlib.sha256(voice_id.encode("utf-8")).hexdigest()[:8], 16)
        base_freq = 170 + (seed % 140)

        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)

            for i in range(samples):
                t = i / sample_rate
                carrier = math.sin(2 * math.pi * base_freq * t)
                envelope = 0.55 + 0.45 * math.sin(2 * math.pi * 2.0 * t)
                value = int(32767 * 0.18 * carrier * envelope)
                wav_file.writeframesraw(value.to_bytes(2, byteorder="little", signed=True))

    def _voice_line_qa(
        self,
        text: str,
        audio_path: str,
        expected_start: float,
        expected_end: float,
        fallback_reason: str,
        fallback_trace: List[str],
        requested_provider: str,
        used_provider: str,
    ) -> Dict[str, Any]:
        expected_duration = max(0.0, float(expected_end) - float(expected_start))
        if expected_duration <= 0:
            expected_duration = self._estimate_speech_duration(text)
        actual_duration = self._audio_duration_seconds(audio_path)
        duration_delta = abs(actual_duration - expected_duration)
        tolerance = max(0.55, expected_duration * 0.35)
        duration_fit = actual_duration <= 0 or duration_delta <= tolerance

        clipping_ratio = self._clipping_ratio(audio_path)
        clipping_detected = clipping_ratio > 0.003

        return {
            "expected_duration_seconds": round(expected_duration, 3),
            "actual_duration_seconds": round(actual_duration, 3),
            "duration_delta_seconds": round(duration_delta, 3),
            "duration_fit": bool(duration_fit),
            "duration_tolerance_seconds": round(tolerance, 3),
            "clipping_ratio": round(clipping_ratio, 6),
            "clipping_detected": clipping_detected,
            "pronunciation_flags": self._pronunciation_flags(text),
            "fallback_reason": fallback_reason,
            "fallback_trace": fallback_trace,
            "requested_provider": requested_provider,
            "used_provider": used_provider,
        }

    @staticmethod
    def _estimate_speech_duration(text: str) -> float:
        words = max(1, len((text or "").split()))
        return max(0.8, min(12.0, words * 0.42))

    def _audio_duration_seconds(self, path: str) -> float:
        if not os.path.exists(path):
            return 0.0
        ext = Path(path).suffix.lower()
        if ext == ".wav":
            try:
                with wave.open(path, "rb") as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate() or 22050
                return float(frames) / float(rate)
            except Exception:
                return 0.0

        ffprobe_path = shutil.which("ffprobe")
        if ffprobe_path:
            cmd = [
                ffprobe_path,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ]
            try:
                completed = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                value = (completed.stdout or "").strip()
                return float(value) if value else 0.0
            except Exception:
                return 0.0
        return 0.0

    def _clipping_ratio(self, path: str) -> float:
        if not os.path.exists(path):
            return 0.0
        if Path(path).suffix.lower() != ".wav":
            return 0.0
        try:
            with wave.open(path, "rb") as wav_file:
                if wav_file.getsampwidth() != 2:
                    return 0.0
                frames = wav_file.readframes(wav_file.getnframes())
            samples = array("h")
            samples.frombytes(frames)
            if not samples:
                return 0.0
            clipped = 0
            for sample in samples:
                if abs(int(sample)) >= 32760:
                    clipped += 1
            return float(clipped) / float(len(samples))
        except Exception:
            return 0.0

    @staticmethod
    def _pronunciation_flags(text: str) -> List[str]:
        flags: List[str] = []
        tokens = [token.strip(".,!?;:()[]{}\"'") for token in (text or "").split()]
        if any(any(char.isdigit() for char in token) for token in tokens):
            flags.append("contains_numbers")
        if any(token.isupper() and len(token) >= 3 for token in tokens):
            flags.append("contains_acronym")
        if any("/" in token or "_" in token for token in tokens):
            flags.append("contains_symbolic_token")
        return flags
