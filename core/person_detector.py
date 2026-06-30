from __future__ import annotations

import time
from pathlib import Path
from typing import Any


class PersonDetector:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", True))
        self.model = None
        self.error: str | None = None
        self.model_path = Path(cfg.get("model_path", "models/yolov8n.pt"))
        self.confidence = float(cfg.get("confidence", 0.45))
        self.imgsz = int(cfg.get("imgsz", 416))
        self.device = str(cfg.get("device", "cuda"))
        self.last_inference_ms: float | None = None

        if self.enabled:
            self._load()

    def _load(self) -> None:
        try:
            from ultralytics import YOLO

            self.model = YOLO(str(self.model_path))
        except Exception as exc:
            self.enabled = False
            self.error = f"YOLO disabled: {exc}"

    def detect(self, frame) -> list[dict[str, Any]]:
        if not self.enabled or self.model is None:
            return []

        started = time.perf_counter()
        results = self.model.predict(
            frame,
            conf=self.confidence,
            classes=[0],
            verbose=False,
            imgsz=self.imgsz,
            device=0 if self.device == "cuda" else self.device,
            half=self.device == "cuda",
        )
        self.last_inference_ms = (time.perf_counter() - started) * 1000
        detections: list[dict[str, Any]] = []
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                detections.append({"box": xyxy, "confidence": float(box.conf[0])})
        return detections

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model_path": str(self.model_path),
            "confidence": self.confidence,
            "imgsz": self.imgsz,
            "device": self.device,
            "last_inference_ms": self.last_inference_ms,
            "error": self.error,
        }
