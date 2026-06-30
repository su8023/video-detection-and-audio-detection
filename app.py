from __future__ import annotations

import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import sounddevice as sd
import yaml
from flask import Flask, Response, jsonify, render_template, request

from core.database import AppDatabase
from core.person_detector import PersonDetector
from core.summarizer import LocalSummarizer
from core.stt_worker import RealtimeTranscriber


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"

VOICE_MODE_PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "pre_roll_ms": 150,
        "endpoint_silence_ms": 250,
        "min_utterance_ms": 350,
        "min_rms": 0.02,
        "description": "低延迟，适合实时字幕，准确率略低。",
    },
    "balanced": {
        "pre_roll_ms": 250,
        "endpoint_silence_ms": 450,
        "min_utterance_ms": 650,
        "min_rms": 0.018,
        "description": "延迟和准确率平衡，推荐日常使用。",
    },
    "accurate": {
        "pre_roll_ms": 300,
        "endpoint_silence_ms": 650,
        "min_utterance_ms": 900,
        "min_rms": 0.018,
        "description": "更完整的句子分段，准确率优先。",
    },
}

FINALIZER_MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "qwen3_0_6b": {
        "finalizer_backend": "qwen3_asr",
        "finalizer_model_id": "Qwen/Qwen3-ASR-0.6B",
        "finalizer_model_size": "0.6B",
        "finalizer_device": "cuda",
        "finalizer_compute_type": "auto",
        "description": "Qwen3-ASR 0.6B，本地 GPU，中文/中英混合优先。",
    },
    "qwen3_1_7b": {
        "finalizer_backend": "qwen3_asr",
        "finalizer_model_id": "Qwen/Qwen3-ASR-1.7B",
        "finalizer_model_size": "1.7B",
        "finalizer_device": "cuda",
        "finalizer_compute_type": "auto",
        "description": "Qwen3-ASR 1.7B，准确率更强，显存和加载时间更高。",
    },
    "whisper_medium": {
        "finalizer_backend": "transformers_cuda",
        "finalizer_model_id": "openai/whisper-medium",
        "finalizer_model_size": "medium",
        "finalizer_device": "cuda",
        "finalizer_compute_type": "float16",
        "description": "OpenAI Whisper medium，本地 GPU，稳定备用。",
    },
    "whisper_small": {
        "finalizer_backend": "transformers_cuda",
        "finalizer_model_id": "openai/whisper-small",
        "finalizer_model_size": "small",
        "finalizer_device": "cuda",
        "finalizer_compute_type": "float16",
        "description": "OpenAI Whisper small，低延迟备用。",
    },
}


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config() -> None:
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)


def detect_voice_mode() -> str:
    stt_cfg = config.get("stt", {})
    for name, preset in VOICE_MODE_PRESETS.items():
        keys = ["pre_roll_ms", "endpoint_silence_ms", "min_utterance_ms"]
        if all(stt_cfg.get(key) == preset.get(key) for key in keys):
            return name
    return "custom"


def detect_finalizer_model() -> str:
    stt_cfg = config.get("stt", {})
    for name, preset in FINALIZER_MODEL_PRESETS.items():
        keys = ["finalizer_backend", "finalizer_model_id"]
        if all(stt_cfg.get(key) == preset.get(key) for key in keys):
            return name
    return "custom"


config = load_config()
app = Flask(__name__)
db_path = ROOT / str(config.get("database", {}).get("path", "data/app.db"))
database = AppDatabase(db_path)
summarizer = LocalSummarizer(config.get("llm", {}))


def generate_summary_from_history(limit: int) -> dict[str, Any]:
    transcripts_for_summary = database.recent_transcripts_for_summary(limit)
    result = summarizer.summarize(transcripts_for_summary)
    source_ids = [item["id"] for item in transcripts_for_summary if item.get("id")]
    summary_id = database.add_summary(
        title=result["title"],
        summary=result["summary"],
        provider=result["provider"],
        model=result["model"],
        transcript_count=len(transcripts_for_summary),
        source_from_id=min(source_ids) if source_ids else None,
        source_to_id=max(source_ids) if source_ids else None,
    )
    result["id"] = summary_id
    result["transcript_count"] = len(transcripts_for_summary)
    return result


class CameraStream:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.lock = threading.Lock()
        self.detection_lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.status: dict[str, Any] = {
            "running": False,
            "opened": False,
            "people_count": 0,
            "last_error": None,
            "frame_width": None,
            "frame_height": None,
            "capture_fps": 0.0,
        }
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.detector_thread: threading.Thread | None = None
        self.detector = PersonDetector(config.get("vision", {}))
        self.last_detections: list[dict[str, Any]] = []

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        if self.detector.enabled:
            self.detector_thread = threading.Thread(target=self._detect_loop, daemon=True)
            self.detector_thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def _backend(self) -> int:
        backend = str(self.cfg.get("backend", "dshow")).lower()
        if backend == "dshow":
            return cv2.CAP_DSHOW
        return cv2.CAP_ANY

    def _capture_loop(self) -> None:
        cam_index = int(self.cfg.get("index", 0))
        cap = cv2.VideoCapture(cam_index, self._backend())
        fourcc = str(self.cfg.get("fourcc", "")).strip().upper()
        if len(fourcc) == 4:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.cfg.get("width", 1280)))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.cfg.get("height", 720)))
        cap.set(cv2.CAP_PROP_FPS, int(self.cfg.get("fps", 30)))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.status.update({"running": True, "opened": bool(cap.isOpened()), "last_error": None})
        if not cap.isOpened():
            self.status["last_error"] = f"Cannot open camera index={cam_index}"
            return

        frames = 0
        fps_started = time.perf_counter()
        while not self.stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                self.status["last_error"] = "Camera read failed"
                time.sleep(0.2)
                continue

            with self.lock:
                self.frame = frame

            frames += 1
            now = time.perf_counter()
            if now - fps_started >= 1.0:
                capture_fps = frames / (now - fps_started)
                frames = 0
                fps_started = now
            else:
                capture_fps = self.status.get("capture_fps", 0.0)

            self.status.update(
                {
                    "opened": True,
                    "frame_width": int(frame.shape[1]),
                    "frame_height": int(frame.shape[0]),
                    "capture_fps": round(float(capture_fps), 1),
                    "people_count": len(self.last_detections),
                    "detector": self.detector.status(),
                }
            )

        cap.release()
        self.status["running"] = False

    def _detect_loop(self) -> None:
        interval = max(0.05, float(config.get("vision", {}).get("detection_interval_ms", 250)) / 1000)
        while not self.stop_event.is_set():
            with self.lock:
                frame = None if self.frame is None else self.frame.copy()

            if frame is None:
                time.sleep(0.05)
                continue

            detections = self.detector.detect(frame)
            with self.detection_lock:
                self.last_detections = detections
            self.status.update(
                {
                    "people_count": len(detections),
                    "detector": self.detector.status(),
                }
            )
            time.sleep(interval)

    def _draw_overlay(self, frame: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            conf = det.get("confidence", 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), (58, 196, 125), 2)
            cv2.putText(
                out,
                f"person {conf:.2f}",
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (58, 196, 125),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            out,
            f"People: {len(detections)}",
            (18, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return out

    def jpeg_frames(self):
        while True:
            with self.lock:
                frame = None if self.frame is None else self.frame.copy()
            with self.detection_lock:
                detections = list(self.last_detections)
            if frame is None:
                placeholder = np.zeros((480, 854, 3), dtype=np.uint8)
                cv2.putText(
                    placeholder,
                    "Waiting for camera...",
                    (40, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    (220, 220, 220),
                    2,
                    cv2.LINE_AA,
                )
                frame = placeholder

            frame = self._draw_overlay(frame, detections)
            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            time.sleep(0.02)


class MonitorAutomation:
    def __init__(self, camera_stream: CameraStream, cfg: dict[str, Any]) -> None:
        self.camera = camera_stream
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", True))
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.last_people_count: int | None = None
        self.absent_since: float | None = None
        self.summary_done_for_absence = False
        self.status: dict[str, Any] = {
            "enabled": self.enabled,
            "last_event": None,
            "absent_since_seconds": None,
            "auto_summary_done": False,
        }

    def start(self) -> None:
        if not self.enabled:
            return
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.status["enabled"] = False

    def apply_config(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", True))
        self.status["enabled"] = self.enabled
        if self.enabled:
            self.start()
        else:
            self.stop()

    def _add_event(self, event_type: str, title: str, detail: str = "", people_count: int | None = None, summary_id: int | None = None) -> None:
        database.add_event(
            event_type=event_type,
            title=title,
            detail=detail,
            people_count=people_count,
            summary_id=summary_id,
        )
        self.status["last_event"] = title

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            if not self.enabled:
                time.sleep(1)
                continue

            interval = max(0.5, float(self.cfg.get("monitor_interval_seconds", 1)))
            absent_confirm = max(1.0, float(self.cfg.get("absent_confirm_seconds", 5)))
            summary_recent_limit = int(self.cfg.get("summary_recent_limit", 80))
            auto_summary_enabled = bool(self.cfg.get("person_absent_summary", True))
            people_count = int(self.camera.status.get("people_count") or 0)
            now = time.perf_counter()

            if self.last_people_count is None:
                self.last_people_count = people_count

            if self.last_people_count == 0 and people_count > 0:
                self.absent_since = None
                self.summary_done_for_absence = False
                self._add_event(
                    "person_enter",
                    f"检测到 {people_count} 人进入画面",
                    people_count=people_count,
                )

            if self.last_people_count > 0 and people_count == 0:
                self.absent_since = now
                self.summary_done_for_absence = False
                self._add_event("person_leave", "检测到人员离开画面", people_count=0)

            if people_count == 0 and self.absent_since is not None:
                absent_seconds = now - self.absent_since
                self.status["absent_since_seconds"] = round(absent_seconds, 1)
                if auto_summary_enabled and not self.summary_done_for_absence and absent_seconds >= absent_confirm:
                    result = generate_summary_from_history(summary_recent_limit)
                    self.summary_done_for_absence = True
                    self._add_event(
                        "auto_summary",
                        "人员离开后自动生成阶段总结",
                        detail=f"{result.get('provider')} / {result.get('model')} / {result.get('transcript_count')} 条记录",
                        people_count=0,
                        summary_id=result.get("id"),
                    )
            else:
                self.status["absent_since_seconds"] = None

            self.status["auto_summary_done"] = self.summary_done_for_absence
            self.last_people_count = people_count
            time.sleep(interval)


def persist_transcript(record: dict[str, Any]) -> None:
    database.add_transcript(record)


camera = CameraStream(config.get("camera", {}))
camera.start()
transcriber = RealtimeTranscriber(config.get("audio", {}), config.get("stt", {}), on_record=persist_transcript)
automation = MonitorAutomation(camera, config.get("automation", {}))
automation.start()


def command_exists(name: str) -> bool:
    try:
        subprocess.run([name, "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except FileNotFoundError:
        return False


def audio_devices() -> list[dict[str, Any]]:
    devices = []
    for index, dev in enumerate(sd.query_devices()):
        devices.append(
            {
                "index": index,
                "name": dev.get("name"),
                "max_input_channels": int(dev.get("max_input_channels", 0)),
                "max_output_channels": int(dev.get("max_output_channels", 0)),
                "default_samplerate": int(dev.get("default_samplerate", 0)),
            }
        )
    return devices


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/video_feed")
def video_feed():
    return Response(camera.jpeg_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/status")
def status():
    return jsonify(
        {
            "app": "local-ai-monitor",
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "camera": camera.status,
            "stt": transcriber.status(),
            "audio_input_devices": [d for d in audio_devices() if d["max_input_channels"] > 0],
            "tools": {
                "ffmpeg": command_exists("ffmpeg"),
                "ollama": command_exists("ollama"),
            },
            "database": {
                "path": str(db_path),
                "transcripts": len(database.list_transcripts(1000)),
                "summaries": len(database.list_summaries(200)),
                "events": len(database.list_events(500)),
            },
            "automation": automation.status,
        }
    )


@app.post("/api/mic-test")
def mic_test():
    payload = request.get_json(silent=True) or {}
    seconds = float(payload.get("seconds", config.get("audio", {}).get("mic_test_seconds", 3)))
    seconds = min(max(seconds, 1.0), 10.0)
    sample_rate = int(config.get("audio", {}).get("sample_rate", 16000))
    channels = int(config.get("audio", {}).get("channels", 1))

    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=channels, dtype="float32")
    sd.wait()
    rms = float(np.sqrt(np.mean(np.square(audio))))
    peak = float(np.max(np.abs(audio)))
    return jsonify({"seconds": seconds, "sample_rate": sample_rate, "channels": channels, "rms": rms, "peak": peak})


@app.post("/api/stt/start")
def stt_start():
    transcriber.start()
    return jsonify(transcriber.status())


@app.post("/api/stt/stop")
def stt_stop():
    transcriber.stop()
    return jsonify(transcriber.status())


@app.post("/api/stt/clear")
def stt_clear():
    transcriber.clear()
    return jsonify({"ok": True, "records": []})


@app.get("/api/transcripts")
def transcripts():
    return jsonify({"status": transcriber.status(), "records": transcriber.transcript()})


@app.get("/api/history")
def history():
    limit = int(request.args.get("limit", 200))
    return jsonify({"records": database.list_transcripts(limit)})


@app.get("/api/history/search")
def history_search():
    query = request.args.get("q", "")
    limit = int(request.args.get("limit", 200))
    return jsonify({"records": database.search_transcripts(query, limit), "query": query})


@app.post("/api/history/clear")
def history_clear():
    database.clear_transcripts()
    return jsonify({"ok": True, "records": []})


@app.get("/api/history/export")
def history_export():
    export_format = request.args.get("format", "md").lower()
    query = request.args.get("q", "")
    limit = int(request.args.get("limit", 1000))
    records = database.search_transcripts(query, limit) if query else database.list_transcripts(limit)
    records = list(reversed(records))

    if export_format == "json":
        import json

        body = json.dumps(records, ensure_ascii=False, indent=2)
        mimetype = "application/json; charset=utf-8"
        filename = "transcripts.json"
    elif export_format == "txt":
        body = "\n".join(
            f"[{item.get('display_time') or item.get('created_at')}] {item.get('speaker') or '说话人'}: {item.get('text')}"
            for item in records
        )
        mimetype = "text/plain; charset=utf-8"
        filename = "transcripts.txt"
    else:
        lines = ["# 语音转写记录", ""]
        if query:
            lines.extend([f"搜索关键词：`{query}`", ""])
        for item in records:
            time_text = item.get("display_time") or item.get("created_at") or ""
            speaker = item.get("speaker") or "说话人"
            text = item.get("text") or ""
            lines.extend([f"## {time_text} {speaker}", "", text, ""])
            if item.get("draft_text") and item.get("draft_text") != text:
                lines.extend([f"> 实时草稿：{item.get('draft_text')}", ""])
        body = "\n".join(lines)
        mimetype = "text/markdown; charset=utf-8"
        filename = "transcripts.md"

    return Response(
        body,
        mimetype=mimetype,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/summaries")
def summaries():
    limit = int(request.args.get("limit", 50))
    return jsonify({"summaries": database.list_summaries(limit)})


@app.post("/api/summary/generate")
def summary_generate():
    payload = request.get_json(silent=True) or {}
    limit = int(payload.get("limit", 80))
    return jsonify(generate_summary_from_history(limit))


@app.get("/api/events")
def events():
    limit = int(request.args.get("limit", 80))
    return jsonify({"events": database.list_events(limit), "automation": automation.status})


@app.post("/api/events/clear")
def events_clear():
    database.clear_events()
    return jsonify({"ok": True, "events": []})


@app.get("/api/settings")
def settings_get():
    return jsonify(
        {
            "voice_mode": detect_voice_mode(),
            "voice_modes": VOICE_MODE_PRESETS,
            "finalizer_model": detect_finalizer_model(),
            "finalizer_models": FINALIZER_MODEL_PRESETS,
            "stt": config.get("stt", {}),
            "automation": config.get("automation", {}),
            "audio_input_devices": [d for d in audio_devices() if d["max_input_channels"] > 0],
        }
    )


@app.post("/api/settings/voice-mode")
def settings_voice_mode():
    payload = request.get_json(silent=True) or {}
    mode = str(payload.get("mode", "balanced"))
    if mode not in VOICE_MODE_PRESETS:
        return jsonify({"ok": False, "error": f"Unknown mode: {mode}"}), 400

    was_running = bool(transcriber.status().get("running"))
    if was_running:
        transcriber.stop()

    preset = VOICE_MODE_PRESETS[mode]
    stt_cfg = config.setdefault("stt", {})
    for key, value in preset.items():
        if key != "description":
            stt_cfg[key] = value
    transcriber.stt_cfg = stt_cfg
    save_config()
    return jsonify({"ok": True, "mode": mode, "stopped_transcriber": was_running, "stt": stt_cfg})


@app.post("/api/settings/asr-model")
def settings_asr_model():
    payload = request.get_json(silent=True) or {}
    model_name = str(payload.get("model", "qwen3_0_6b"))
    if model_name not in FINALIZER_MODEL_PRESETS:
        return jsonify({"ok": False, "error": f"Unknown ASR model: {model_name}"}), 400

    was_running = bool(transcriber.status().get("running"))
    if was_running:
        transcriber.stop()

    preset = FINALIZER_MODEL_PRESETS[model_name]
    stt_cfg = config.setdefault("stt", {})
    for key, value in preset.items():
        if key != "description":
            stt_cfg[key] = value
    transcriber.stt_cfg = stt_cfg
    transcriber.whisper_model = None
    transcriber.qwen_asr_model = None
    transcriber.transformers_model = None
    transcriber.transformers_processor = None
    save_config()
    return jsonify({"ok": True, "model": model_name, "stopped_transcriber": was_running, "stt": stt_cfg})


@app.post("/api/settings/automation")
def settings_automation():
    payload = request.get_json(silent=True) or {}
    automation_cfg = config.setdefault("automation", {})
    for key in ["enabled", "person_absent_summary"]:
        if key in payload:
            automation_cfg[key] = bool(payload[key])
    for key in ["absent_confirm_seconds", "summary_recent_limit", "monitor_interval_seconds"]:
        if key in payload:
            automation_cfg[key] = int(payload[key])
    automation.apply_config(automation_cfg)
    save_config()
    return jsonify({"ok": True, "automation": automation_cfg})


if __name__ == "__main__":
    host = config.get("app", {}).get("host", "127.0.0.1")
    port = int(config.get("app", {}).get("port", 5050))
    app.run(host=host, port=port, debug=bool(config.get("app", {}).get("debug", False)), threaded=True)
