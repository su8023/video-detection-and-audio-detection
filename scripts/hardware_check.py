from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import cv2
import sounddevice as sd


def check_camera() -> None:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    opened = cap.isOpened()
    print(f"camera.opened={opened}")
    if opened:
        ok, frame = cap.read()
        print(f"camera.frame={ok}")
        if ok:
            print(f"camera.shape={frame.shape[1]}x{frame.shape[0]}")
    cap.release()


def check_audio() -> None:
    print("audio.input_devices=")
    for index, dev in enumerate(sd.query_devices()):
        if int(dev.get("max_input_channels", 0)) > 0:
            print(f"  #{index} {dev.get('name')} ({int(dev.get('default_samplerate', 0))} Hz)")


def check_python_modules() -> None:
    for name in ["torch", "ultralytics", "faster_whisper"]:
        print(f"module.{name}={importlib.util.find_spec(name) is not None}")

    if importlib.util.find_spec("torch"):
        import torch

        print(f"torch.cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"torch.cuda.device={torch.cuda.get_device_name(0)}")


def main() -> None:
    print(f"python={sys.version.split()[0]}")
    print(f"cwd={Path.cwd()}")
    print(f"ffmpeg={shutil.which('ffmpeg') or 'not-found'}")
    print(f"ollama={shutil.which('ollama') or 'not-found'}")
    check_python_modules()
    check_camera()
    check_audio()


if __name__ == "__main__":
    main()
