from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

VOSK_MODEL_NAME = "vosk-model-small-cn-0.22"
VOSK_URL = f"https://alphacephei.com/vosk/models/{VOSK_MODEL_NAME}.zip"
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt"

ASR_REPOS = {
    "qwen3_0_6b": "Qwen/Qwen3-ASR-0.6B",
    "qwen3_1_7b": "Qwen/Qwen3-ASR-1.7B",
    "whisper_medium": "openai/whisper-medium",
    "whisper_small": "openai/whisper-small",
}


def download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        print(f"[skip] {target} already exists")
        return

    tmp = target.with_suffix(target.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    def report(block_count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(block_count * block_size, total_size)
        pct = downloaded * 100 / total_size
        print(f"\r[download] {target.name} {pct:5.1f}%", end="")

    print(f"[download] {url}")
    urllib.request.urlretrieve(url, tmp, reporthook=report)
    print()
    tmp.replace(target)


def prepare_vosk() -> None:
    model_dir = MODELS_DIR / VOSK_MODEL_NAME
    if (model_dir / "am" / "final.mdl").exists():
        print(f"[skip] Vosk model ready: {model_dir}")
        return

    zip_path = MODELS_DIR / f"{VOSK_MODEL_NAME}.zip"
    download_file(VOSK_URL, zip_path)
    print(f"[extract] {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(MODELS_DIR)
    print(f"[ok] Vosk model ready: {model_dir}")


def prepare_yolo() -> None:
    download_file(YOLO_URL, MODELS_DIR / "yolov8n.pt")


def prepare_asr(names: list[str]) -> None:
    if not names:
        return
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required. Install requirements-ai.txt first.") from exc

    selected = list(ASR_REPOS) if "all" in names else names
    for name in selected:
        repo_id = ASR_REPOS[name]
        print(f"[hf] downloading/cache warming: {repo_id}")
        path = snapshot_download(repo_id=repo_id)
        print(f"[ok] {repo_id} -> {path}")


def clean_local_models() -> None:
    if MODELS_DIR.exists():
        print(f"[remove] {MODELS_DIR}")
        shutil.rmtree(MODELS_DIR)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / ".gitkeep").touch()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download runtime model files for Local AI Monitor.")
    parser.add_argument("--skip-core", action="store_true", help="Do not download Vosk and YOLO models.")
    parser.add_argument(
        "--asr",
        nargs="*",
        choices=[*ASR_REPOS.keys(), "all"],
        default=[],
        help="Optionally pre-download ASR models into the Hugging Face cache.",
    )
    parser.add_argument("--clean", action="store_true", help="Remove local models directory before downloading.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.clean:
        clean_local_models()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_core:
        prepare_yolo()
        prepare_vosk()

    prepare_asr(args.asr)
    print("[done] runtime models are ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
