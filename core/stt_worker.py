from __future__ import annotations

import json
import queue
import re
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd


class RealtimeTranscriber:
    def __init__(self, audio_cfg: dict[str, Any], stt_cfg: dict[str, Any], on_record=None) -> None:
        self.audio_cfg = audio_cfg
        self.stt_cfg = stt_cfg
        self.on_record = on_record
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.vosk_model = None
        self.whisper_model = None
        self.qwen_asr_model = None
        self.transformers_model = None
        self.transformers_processor = None
        self.transformers_device = None
        self.transformers_dtype = None
        self.finalizer_thread: threading.Thread | None = None
        self.finalizer_queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=20)
        self.records: deque[dict[str, Any]] = deque(maxlen=200)
        self.partial_text = ""
        self.status_data: dict[str, Any] = {
            "running": False,
            "loading": False,
            "last_error": None,
            "last_rms": 0.0,
            "last_peak": 0.0,
            "engine": stt_cfg.get("engine", "vosk_stream"),
            "model_size": stt_cfg.get("model_size", "base"),
            "device": stt_cfg.get("device", "cpu"),
            "chunk_seconds": stt_cfg.get("chunk_seconds", 4),
            "partial": "",
            "partial_age_ms": None,
            "last_decode_ms": None,
            "finalizer_enabled": bool(stt_cfg.get("finalizer_enabled", True)),
            "finalizer_backend": stt_cfg.get("finalizer_backend", "faster_whisper"),
            "finalizer_model_id": stt_cfg.get("finalizer_model_id"),
            "finalizer_model_size": stt_cfg.get("finalizer_model_size", "small"),
            "finalizer_loading": False,
            "finalizer_pending": 0,
            "finalizer_last_ms": None,
            "finalizer_language": None,
            "endpoint_silence_ms": stt_cfg.get("endpoint_silence_ms", 350),
            "pre_roll_ms": stt_cfg.get("pre_roll_ms", 0),
            "last_endpoint_delay_ms": None,
            "records_count": 0,
        }
        self.last_partial_at: float | None = None

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        with self.lock:
            self.status_data.update({"running": True, "loading": True, "last_error": None})
        self.stop_event.clear()
        if self._finalizer_enabled() and not (self.finalizer_thread and self.finalizer_thread.is_alive()):
            self.finalizer_thread = threading.Thread(target=self._finalizer_loop, daemon=True)
            self.finalizer_thread.start()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        with self.lock:
            self.status_data["running"] = False

    def clear(self) -> None:
        while True:
            try:
                self.finalizer_queue.get_nowait()
            except queue.Empty:
                break
        with self.lock:
            self.records.clear()
            self.partial_text = ""
            self.status_data.update(
                {"records_count": 0, "partial": "", "partial_age_ms": None, "finalizer_pending": 0}
            )

    def status(self) -> dict[str, Any]:
        with self.lock:
            data = dict(self.status_data)
            if self.last_partial_at:
                data["partial_age_ms"] = int((time.perf_counter() - self.last_partial_at) * 1000)
            return data

    def transcript(self) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.records)

    def _loop(self) -> None:
        engine = str(self.stt_cfg.get("engine", "vosk_stream"))
        if engine == "vosk_stream":
            self._vosk_stream_loop()
        else:
            self._whisper_chunk_loop()

    def _add_record(
        self,
        text: str,
        elapsed_ms: float | None = None,
        *,
        engine: str | None = None,
        language: str | None = None,
        draft_text: str | None = None,
    ) -> None:
        text = text.strip()
        if not text:
            return
        min_record_chars = int(self.stt_cfg.get("min_record_chars", 1))
        if len(text) < min_record_chars:
            return
        now = datetime.now()
        record = {
            "id": f"{now.timestamp():.3f}",
            "time": now.strftime("%H:%M:%S"),
            "speaker": "说话人 A",
            "text": text,
            "elapsed_ms": elapsed_ms,
            "engine": engine,
            "language": language,
            "draft_text": draft_text,
        }
        with self.lock:
            self.records.append(record)
            self.partial_text = ""
            self.status_data.update(
                {"records_count": len(self.records), "partial": "", "partial_age_ms": None}
            )
        if self.on_record:
            try:
                self.on_record(record)
            except Exception as exc:
                with self.lock:
                    self.status_data["last_error"] = f"persist: {exc}"

    def _add_pending_record(self, draft_text: str, started_at: datetime) -> str:
        text = draft_text.strip() or "正在识别..."
        record_id = f"{started_at.timestamp():.3f}"
        record = {
            "id": record_id,
            "time": started_at.strftime("%H:%M:%S"),
            "speaker": "说话人 A",
            "text": text,
            "elapsed_ms": None,
            "engine": "vosk+whisper",
            "language": None,
            "draft_text": draft_text,
            "pending": True,
        }
        with self.lock:
            self.records.append(record)
            self.partial_text = ""
            self.status_data.update(
                {"records_count": len(self.records), "partial": "", "partial_age_ms": None}
            )
        return record_id

    def _remove_record(self, record_id: str | None) -> None:
        if not record_id:
            return
        with self.lock:
            self.records = deque((record for record in self.records if record.get("id") != record_id), maxlen=200)
            self.status_data["records_count"] = len(self.records)

    def _finalize_record(
        self,
        record_id: str | None,
        text: str,
        elapsed_ms: float | None = None,
        *,
        engine: str | None = None,
        language: str | None = None,
        draft_text: str | None = None,
    ) -> None:
        text = text.strip()
        if not text:
            self._remove_record(record_id)
            return

        min_record_chars = int(self.stt_cfg.get("min_record_chars", 1))
        if len(text) < min_record_chars:
            self._remove_record(record_id)
            return

        finalized_record: dict[str, Any] | None = None
        with self.lock:
            for record in self.records:
                if record.get("id") == record_id:
                    record.update(
                        {
                            "text": text,
                            "elapsed_ms": elapsed_ms,
                            "engine": engine,
                            "language": language,
                            "draft_text": draft_text,
                            "pending": False,
                        }
                    )
                    finalized_record = dict(record)
                    break
            self.partial_text = ""
            self.status_data.update(
                {"records_count": len(self.records), "partial": "", "partial_age_ms": None}
            )

        if finalized_record is None:
            self._add_record(text, elapsed_ms, engine=engine, language=language, draft_text=draft_text)
            return

        if self.on_record:
            try:
                self.on_record(finalized_record)
            except Exception as exc:
                with self.lock:
                    self.status_data["last_error"] = f"persist: {exc}"

    def _input_device(self) -> int | None:
        input_device = self.audio_cfg.get("input_device")
        return None if input_device in ("", "null", None) else int(input_device)

    def _finalizer_enabled(self) -> bool:
        return bool(self.stt_cfg.get("finalizer_enabled", True))

    def _queue_finalizer(self, audio_bytes: bytes, draft_text: str, started_at: datetime) -> None:
        if not audio_bytes:
            self._add_record(draft_text, engine="vosk", draft_text=draft_text)
            return
        record_id = self._add_pending_record(draft_text, started_at)
        item = {"audio_bytes": audio_bytes, "draft_text": draft_text, "started_at": started_at}
        item["record_id"] = record_id
        try:
            self.finalizer_queue.put_nowait(item)
            with self.lock:
                self.status_data["finalizer_pending"] = self.finalizer_queue.qsize()
        except queue.Full:
            self._finalize_record(record_id, draft_text, engine="vosk", draft_text=draft_text)

    def _load_whisper_finalizer(self) -> None:
        if self.whisper_model is not None:
            return
        from faster_whisper import WhisperModel

        model_size = str(self.stt_cfg.get("finalizer_model_size", "small"))
        device = str(self.stt_cfg.get("finalizer_device", "cpu"))
        compute_type = str(self.stt_cfg.get("finalizer_compute_type", "int8"))
        with self.lock:
            self.status_data["finalizer_loading"] = True
        self.whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        with self.lock:
            self.status_data["finalizer_loading"] = False

    def _load_transformers_finalizer(self) -> None:
        if self.transformers_model is not None and self.transformers_processor is not None:
            return

        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        model_id = str(self.stt_cfg.get("finalizer_model_id", "openai/whisper-medium"))
        requested_device = str(self.stt_cfg.get("finalizer_device", "cuda"))
        device = "cuda:0" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"
        compute_type = str(self.stt_cfg.get("finalizer_compute_type", "float16"))
        dtype = torch.float16 if device.startswith("cuda") and compute_type == "float16" else torch.float32

        with self.lock:
            self.status_data["finalizer_loading"] = True

        self.transformers_processor = AutoProcessor.from_pretrained(model_id)
        self.transformers_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(device)
        self.transformers_model.eval()
        self.transformers_device = device
        self.transformers_dtype = dtype

        with self.lock:
            self.status_data.update(
                {
                    "finalizer_loading": False,
                    "finalizer_model_id": model_id,
                    "finalizer_device": device,
                }
            )

    def _torch_dtype_from_config(self):
        import torch

        requested_device = str(self.stt_cfg.get("finalizer_device", "cuda"))
        device = "cuda:0" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"
        compute_type = str(self.stt_cfg.get("finalizer_compute_type", "auto")).lower()
        if device.startswith("cuda") and compute_type in ("auto", "bf16", "bfloat16") and torch.cuda.is_bf16_supported():
            return device, torch.bfloat16
        if device.startswith("cuda") and compute_type in ("auto", "fp16", "float16", "half"):
            return device, torch.float16
        return device, torch.float32

    def _load_qwen_finalizer(self) -> None:
        if self.qwen_asr_model is not None:
            return

        from qwen_asr import Qwen3ASRModel

        model_id = str(self.stt_cfg.get("finalizer_model_id", "Qwen/Qwen3-ASR-0.6B"))
        max_inference_batch_size = int(self.stt_cfg.get("qwen_max_inference_batch_size", 4))
        max_new_tokens = int(self.stt_cfg.get("qwen_max_new_tokens", 128))
        device, dtype = self._torch_dtype_from_config()

        with self.lock:
            self.status_data["finalizer_loading"] = True

        self.qwen_asr_model = Qwen3ASRModel.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
        )

        with self.lock:
            self.status_data.update(
                {
                    "finalizer_loading": False,
                    "finalizer_model_id": model_id,
                    "finalizer_device": device,
                }
            )

    def _transcribe_with_transformers(self, audio: np.ndarray, sample_rate: int, language: str | None) -> tuple[str, str | None]:
        import torch

        self._load_transformers_finalizer()
        processor = self.transformers_processor
        model = self.transformers_model
        device = self.transformers_device or "cpu"
        dtype = self.transformers_dtype or torch.float32

        inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        input_features = inputs.input_features.to(device=device, dtype=dtype)
        generate_kwargs: dict[str, Any] = {
            "task": "transcribe",
            "num_beams": 1,
            "do_sample": False,
            "max_new_tokens": 96,
            "use_cache": True,
        }
        if language:
            generate_kwargs["language"] = language

        with torch.inference_mode():
            predicted_ids = model.generate(input_features, **generate_kwargs)
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        return text, language

    def _transcribe_with_qwen(self, audio: np.ndarray, sample_rate: int, language: str | None) -> tuple[str, str | None]:
        self._load_qwen_finalizer()
        results = self.qwen_asr_model.transcribe(audio=(audio, sample_rate), language=language)
        if not results:
            return "", language
        result = results[0]
        return (getattr(result, "text", "") or "").strip(), getattr(result, "language", None) or language

    def _finalizer_loop(self) -> None:
        sample_rate = int(self.audio_cfg.get("sample_rate", 16000))
        backend = str(self.stt_cfg.get("finalizer_backend", "faster_whisper"))
        language = self.stt_cfg.get("finalizer_language")
        language = None if language in ("", "null", None) else str(language)
        no_speech_threshold = float(self.stt_cfg.get("no_speech_threshold", 0.65))
        log_prob_threshold = float(self.stt_cfg.get("log_prob_threshold", -1.0))

        if backend in ("transformers_cuda", "qwen3_asr") and bool(self.stt_cfg.get("finalizer_preload", True)):
            try:
                if backend == "qwen3_asr":
                    self._load_qwen_finalizer()
                else:
                    self._load_transformers_finalizer()
            except Exception as exc:
                with self.lock:
                    self.status_data.update({"last_error": f"finalizer preload: {exc}", "finalizer_loading": False})

        while True:
            item = self.finalizer_queue.get()
            if item is None:
                return

            draft_text = str(item.get("draft_text") or "")
            audio_bytes = bytes(item.get("audio_bytes") or b"")
            record_id = item.get("record_id")
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            audio_seconds = audio.size / sample_rate if sample_rate else 0
            if audio.size < sample_rate * 0.25:
                self._finalize_record(record_id, draft_text, engine="vosk", draft_text=draft_text)
                continue

            try:
                started = time.perf_counter()
                if backend == "qwen3_asr":
                    text, detected_language = self._transcribe_with_qwen(audio, sample_rate, language)
                elif backend == "transformers_cuda":
                    text, detected_language = self._transcribe_with_transformers(audio, sample_rate, language)
                else:
                    self._load_whisper_finalizer()
                    segments, info = self.whisper_model.transcribe(
                        audio,
                        language=language,
                        vad_filter=True,
                        beam_size=3,
                        temperature=0.0,
                        condition_on_previous_text=False,
                        no_speech_threshold=no_speech_threshold,
                        log_prob_threshold=log_prob_threshold,
                        compression_ratio_threshold=2.4,
                    )
                    text_parts = []
                    for segment in segments:
                        no_speech_prob = float(getattr(segment, "no_speech_prob", 0.0))
                        avg_logprob = float(getattr(segment, "avg_logprob", 0.0))
                        if no_speech_prob > no_speech_threshold or avg_logprob < log_prob_threshold:
                            continue
                        text_parts.append(segment.text.strip())
                    text = " ".join(text_parts).strip()
                    detected_language = getattr(info, "language", None)

                elapsed_ms = (time.perf_counter() - started) * 1000
                text = text.strip() or draft_text
                if self._looks_like_short_hallucination(text, draft_text, audio_seconds):
                    if draft_text.strip():
                        self._finalize_record(record_id, draft_text, engine="vosk", draft_text=draft_text)
                    else:
                        self._remove_record(record_id)
                    with self.lock:
                        self.status_data.update(
                            {
                                "finalizer_last_ms": elapsed_ms,
                                "finalizer_pending": self.finalizer_queue.qsize(),
                            }
                        )
                    continue
                self._finalize_record(
                    record_id,
                    text,
                    elapsed_ms,
                    engine="qwen3-asr" if backend == "qwen3_asr" else ("whisper-gpu" if backend == "transformers_cuda" else "whisper"),
                    language=detected_language,
                    draft_text=draft_text,
                )
                with self.lock:
                    self.status_data.update(
                        {
                            "finalizer_last_ms": elapsed_ms,
                            "finalizer_language": detected_language,
                            "finalizer_pending": self.finalizer_queue.qsize(),
                        }
                    )
            except Exception as exc:
                with self.lock:
                    self.status_data.update(
                        {
                            "last_error": f"finalizer: {exc}",
                            "finalizer_loading": False,
                            "finalizer_pending": self.finalizer_queue.qsize(),
                        }
                    )
                self._finalize_record(record_id, draft_text, engine="vosk", draft_text=draft_text)

    def _looks_like_short_hallucination(self, text: str, draft_text: str, audio_seconds: float) -> bool:
        normalized = text.strip().lower().strip(".!?。！？")
        if not normalized:
            return True
        common_noise = {"bye", "hello", "hi", "you", "okay", "ok", "thanks", "thank you", "hmm", "um", "ah"}
        if audio_seconds < 1.2 and normalized in common_noise and not draft_text.strip():
            return True
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", text))
        has_latin = bool(re.search(r"[A-Za-z]", text))
        if audio_seconds < 0.9 and has_latin and not has_cjk and len(normalized) <= 12 and not draft_text.strip():
            return True
        return False

    def _vosk_stream_loop(self) -> None:
        sample_rate = int(self.audio_cfg.get("sample_rate", 16000))
        model_path = Path(str(self.stt_cfg.get("vosk_model_path", "models/vosk-model-small-cn-0.22")))
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parents[1] / model_path
        partial_interval = max(0.05, float(self.stt_cfg.get("partial_interval_ms", 200)) / 1000)
        stream_block_ms = max(20, int(self.stt_cfg.get("stream_block_ms", 50)))
        blocksize = max(320, int(sample_rate * stream_block_ms / 1000))
        pre_roll_ms = max(0, int(self.stt_cfg.get("pre_roll_ms", 0)))
        pre_roll_blocks = max(0, int(pre_roll_ms / stream_block_ms))
        endpoint_silence = max(0.1, float(self.stt_cfg.get("endpoint_silence_ms", 350)) / 1000)
        min_utterance = max(0.1, float(self.stt_cfg.get("min_utterance_ms", 350)) / 1000)
        max_utterance = max(1.0, float(self.stt_cfg.get("max_utterance_seconds", 12)))
        min_rms = float(self.stt_cfg.get("min_rms", 0.006))
        input_device = self._input_device()
        audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=30)

        with self.lock:
            self.status_data.update({"running": True, "loading": True, "last_error": None})

        try:
            from vosk import KaldiRecognizer, Model, SetLogLevel

            SetLogLevel(-1)
            self.vosk_model = self.vosk_model or Model(str(model_path))
            recognizer = KaldiRecognizer(self.vosk_model, sample_rate)
            recognizer.SetWords(False)
        except Exception as exc:
            with self.lock:
                self.status_data.update({"running": False, "loading": False, "last_error": str(exc)})
            return

        def callback(indata, frames, time_info, status) -> None:
            if status:
                with self.lock:
                    self.status_data["last_error"] = str(status)
            try:
                audio_queue.put_nowait(bytes(indata))
            except queue.Full:
                pass

        with self.lock:
            self.status_data["loading"] = False

        try:
            with sd.RawInputStream(
                samplerate=sample_rate,
                blocksize=blocksize,
                device=input_device,
                dtype="int16",
                channels=1,
                callback=callback,
            ):
                last_partial_emit = 0.0
                utterance_audio = bytearray()
                utterance_started_at = datetime.now()
                utterance_started_perf: float | None = None
                last_voice_at: float | None = None
                pre_roll: deque[bytes] = deque(maxlen=pre_roll_blocks)

                def flush_utterance(draft_text: str, decode_ms: float | None = None) -> None:
                    nonlocal utterance_audio, utterance_started_at, utterance_started_perf, last_voice_at
                    if not utterance_audio or utterance_started_perf is None:
                        return
                    duration = time.perf_counter() - utterance_started_perf
                    if duration < min_utterance:
                        utterance_audio = bytearray()
                        utterance_started_perf = None
                        last_voice_at = None
                        recognizer.Reset()
                        return
                    if self._finalizer_enabled():
                        self._queue_finalizer(bytes(utterance_audio), draft_text, utterance_started_at)
                    else:
                        self._add_record(draft_text, decode_ms, engine="vosk", draft_text=draft_text)
                    endpoint_delay = None
                    if last_voice_at is not None:
                        endpoint_delay = (time.perf_counter() - last_voice_at) * 1000
                    utterance_audio = bytearray()
                    utterance_started_at = datetime.now()
                    utterance_started_perf = None
                    last_voice_at = None
                    recognizer.Reset()
                    with self.lock:
                        self.status_data.update(
                            {
                                "last_decode_ms": decode_ms,
                                "last_endpoint_delay_ms": endpoint_delay,
                                "finalizer_pending": self.finalizer_queue.qsize(),
                                "partial": "",
                                "partial_age_ms": None,
                            }
                        )

                while not self.stop_event.is_set():
                    try:
                        data = audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
                    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
                    now = time.perf_counter()
                    is_voice = rms >= min_rms
                    if is_voice and not utterance_audio:
                        utterance_started_at = datetime.now()
                        utterance_started_perf = now
                        for chunk in pre_roll:
                            utterance_audio.extend(chunk)
                    if is_voice:
                        last_voice_at = now
                    if is_voice or utterance_audio:
                        utterance_audio.extend(data)
                    elif pre_roll_blocks:
                        pre_roll.append(data)
                    with self.lock:
                        self.status_data.update({"last_rms": rms, "last_peak": peak})

                    started = time.perf_counter()
                    is_final = recognizer.AcceptWaveform(data)
                    decode_ms = (time.perf_counter() - started) * 1000

                    if is_final:
                        result = json.loads(recognizer.Result() or "{}")
                        text = (result.get("text") or "").replace(" ", "")
                        flush_utterance(text, decode_ms)
                        continue

                    if now - last_partial_emit >= partial_interval:
                        partial = json.loads(recognizer.PartialResult() or "{}").get("partial", "")
                        partial = partial.replace(" ", "")
                        if partial or rms >= min_rms:
                            self.last_partial_at = now
                            with self.lock:
                                self.partial_text = partial
                                self.status_data.update(
                                    {
                                        "partial": partial,
                                        "partial_age_ms": 0,
                                        "last_decode_ms": decode_ms,
                                    }
                                )
                        last_partial_emit = now

                    if utterance_audio and last_voice_at is not None:
                        draft = json.loads(recognizer.PartialResult() or "{}").get("partial", "").replace(" ", "")
                        silence = now - last_voice_at
                        duration = 0 if utterance_started_perf is None else now - utterance_started_perf
                        if silence >= endpoint_silence or duration >= max_utterance:
                            flush_utterance(draft, decode_ms)
        except Exception as exc:
            with self.lock:
                self.status_data["last_error"] = str(exc)

        final = ""
        try:
            final = json.loads(recognizer.FinalResult() or "{}").get("text", "").replace(" ", "")
        except Exception:
            final = ""
        if final:
            self._add_record(final, engine="vosk", draft_text=final)
        with self.lock:
            self.status_data["running"] = False

    def _load_whisper_model(self) -> None:
        if self.whisper_model is not None:
            return
        from faster_whisper import WhisperModel

        model_size = str(self.stt_cfg.get("model_size", "base"))
        device = str(self.stt_cfg.get("device", "cpu"))
        compute_type = str(self.stt_cfg.get("compute_type", "int8"))
        self.whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def _whisper_chunk_loop(self) -> None:
        sample_rate = int(self.audio_cfg.get("sample_rate", 16000))
        channels = int(self.audio_cfg.get("channels", 1))
        chunk_seconds = float(self.stt_cfg.get("chunk_seconds", 4))
        min_rms = float(self.stt_cfg.get("min_rms", 0.006))
        language = self.stt_cfg.get("language")
        language = None if language in ("", "null", None) else str(language)
        no_speech_threshold = float(self.stt_cfg.get("no_speech_threshold", 0.65))
        log_prob_threshold = float(self.stt_cfg.get("log_prob_threshold", -1.0))
        input_device = self._input_device()

        with self.lock:
            self.status_data.update({"running": True, "loading": True, "last_error": None})

        try:
            self._load_whisper_model()
        except Exception as exc:
            with self.lock:
                self.status_data.update({"running": False, "loading": False, "last_error": str(exc)})
            return

        with self.lock:
            self.status_data["loading"] = False

        while not self.stop_event.is_set():
            try:
                audio = sd.rec(
                    int(chunk_seconds * sample_rate),
                    samplerate=sample_rate,
                    channels=channels,
                    dtype="float32",
                    device=input_device,
                )
                sd.wait()
                audio = np.squeeze(audio)
                rms = float(np.sqrt(np.mean(np.square(audio))))
                peak = float(np.max(np.abs(audio)))
                with self.lock:
                    self.status_data.update({"last_rms": rms, "last_peak": peak})
                if rms < min_rms:
                    continue

                started = time.perf_counter()
                segments, _info = self.whisper_model.transcribe(
                    audio,
                    language=language,
                    vad_filter=True,
                    beam_size=1,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    no_speech_threshold=no_speech_threshold,
                    log_prob_threshold=log_prob_threshold,
                    compression_ratio_threshold=2.4,
                )
                text_parts = []
                for segment in segments:
                    no_speech_prob = float(getattr(segment, "no_speech_prob", 0.0))
                    avg_logprob = float(getattr(segment, "avg_logprob", 0.0))
                    if no_speech_prob > no_speech_threshold or avg_logprob < log_prob_threshold:
                        continue
                    text_parts.append(segment.text.strip())
                elapsed_ms = (time.perf_counter() - started) * 1000
                self._add_record("".join(text_parts), elapsed_ms)
                with self.lock:
                    self.status_data["last_decode_ms"] = elapsed_ms
            except Exception as exc:
                with self.lock:
                    self.status_data["last_error"] = str(exc)
                time.sleep(0.5)

        with self.lock:
            self.status_data["running"] = False
