from __future__ import annotations

import argparse
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Record microphone audio and transcribe it locally.")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--model", default="small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--language", default="zh")
    args = parser.parse_args()

    sample_rate = 16000
    print(f"recording {args.seconds:.1f}s, please speak now...")
    audio = sd.rec(int(args.seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    audio = np.squeeze(audio)
    rms = float(np.sqrt(np.mean(np.square(audio))))
    peak = float(np.max(np.abs(audio)))
    print(f"audio.rms={rms:.5f} audio.peak={peak:.5f}")

    started = time.time()
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    segments, info = model.transcribe(audio, language=args.language, vad_filter=True)
    print(f"language={info.language} probability={info.language_probability:.2f}")
    text_parts = []
    for segment in segments:
        line = f"[{segment.start:.2f}-{segment.end:.2f}] {segment.text.strip()}"
        print(line)
        text_parts.append(segment.text.strip())
    print(f"text={''.join(text_parts) or '(empty)'}")
    print(f"elapsed={time.time() - started:.2f}s")


if __name__ == "__main__":
    main()
