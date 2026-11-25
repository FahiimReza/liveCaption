import ctypes
from difflib import SequenceMatcher
import gc
import logging
import os
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
import numpy as np
import pyaudiowpatch as pa

# --- Project-Local Model Storage & Hub Environment ---
# Must be set before importing faster_whisper / huggingface_hub so model weights,
# VAD assets, and Hugging Face caches stay inside <project>/models without symlink warnings.
PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(MODELS_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODELS_DIR / "hub"))
os.environ.setdefault("HF_HUB_CACHE", str(MODELS_DIR / "hub"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from faster_whisper import WhisperModel

try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("live_caption")

DEFAULT_MODEL_SIZE = "large-v3-turbo"
MODEL_CHOICES = ["small", "medium", "large-v3-turbo"]
TRANSLATE_CHOICES = ["none", "to_en", "target"]

# "auto" lets Whisper auto-detect; setting a code (e.g. "en", "bn") locks the source language
DEFAULT_SOURCE_LANG = "auto"
SOURCE_LANG_CHOICES = ["auto", "en", "bn", "hi", "es", "fr", "ja"]

DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "float16"

SAMPLE_RATE = 16000
CHUNK_SECONDS = 0.5
WINDOW_SECONDS = 3.0
INFER_INTERVAL_SECONDS = 0.45
NO_REPEAT_NGRAM_SIZE = 3
SILENCE_RMS_THRESHOLD = 0.012
SILENCE_RESET_CHUNKS = 4  # ~2.0s of low-energy chunks before clearing stale stabilizer state
IDLE_CLEAR_SECONDS = 1.2  # Clear displayed captions after this long with no new speech
TRANSLATION_HOLD_SECONDS = 5.0  # Keep a late-arriving translation visible this long
MAX_CONFIRMED_WORDS = 28
MAX_SUBTITLE_WORDS = 11  # Total words shown at once in the subtitle stream
LOG_CHUNK_EVERY = 10

# VAD tuning: lower threshold (e.g., 0.30-0.40) retains softer voices/trailing syllables
VAD_THRESHOLD = 0.35
VAD_MIN_SILENCE_MS = 400

# Set to True only if you have plenty of VRAM and want instant switching back to prior models.
CACHE_LOADED_MODELS = False

# Translation settings:
# - "none"   -> keep transcribed language
# - "to_en"  -> Whisper native translation to English
# - "target" -> translate finalized lines to TARGET_LANG via GoogleTranslator
DEFAULT_TRANSLATE_MODE = "none"
TARGET_LANG = "bn"  # e.g., "bn", "es", "fr", "de", "ja"


def _enable_windows_dpi_awareness():
    """Prevents Windows bitmap upscaling/blurriness on Tk windows."""
    if os.name != "nt":
        return
    try:
        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return
        shcore = getattr(windll, "shcore", None)
        if shcore is not None:
            try:
                # PROCESS_PER_MONITOR_DPI_AWARE = 2
                shcore.SetProcessDpiAwareness(2)
                return
            except Exception:
                pass
        windll.user32.SetProcessDPIAware()
    except Exception as exc:
        logger.debug("Could not set DPI awareness: %s", exc)


class RuntimeConfig:
    """Thread-safe runtime state shared between the overlay UI and ASR worker."""

    def __init__(self, model_size: str, translate_mode: str, source_lang: str, target_lang: str):
        self._lock = threading.Lock()
        self.model_size = model_size
        self.translate_mode = translate_mode
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.model_reload_event = threading.Event()

    def set_translate_mode(self, mode: str) -> str:
        with self._lock:
            if mode in TRANSLATE_CHOICES:
                self.translate_mode = mode
            return self.translate_mode

    def cycle_translate_mode(self) -> str:
        with self._lock:
            idx = TRANSLATE_CHOICES.index(self.translate_mode) if self.translate_mode in TRANSLATE_CHOICES else 0
            self.translate_mode = TRANSLATE_CHOICES[(idx + 1) % len(TRANSLATE_CHOICES)]
            return self.translate_mode

    def cycle_source_lang(self) -> str:
        with self._lock:
            idx = (
                SOURCE_LANG_CHOICES.index(self.source_lang)
                if self.source_lang in SOURCE_LANG_CHOICES
                else 0
            )
            self.source_lang = SOURCE_LANG_CHOICES[(idx + 1) % len(SOURCE_LANG_CHOICES)]
            return self.source_lang

    def set_model_size(self, size: str) -> str:
        with self._lock:
            if size != self.model_size:
                self.model_size = size
                self.model_reload_event.set()
            return self.model_size

    def cycle_model_size(self) -> str:
        with self._lock:
            idx = MODEL_CHOICES.index(self.model_size) if self.model_size in MODEL_CHOICES else 0
            self.model_size = MODEL_CHOICES[(idx + 1) % len(MODEL_CHOICES)]
            self.model_reload_event.set()
            return self.model_size

    def snapshot(self) -> tuple[str, str, str, str]:
        with self._lock:
            return self.model_size, self.translate_mode, self.source_lang, self.target_lang


class LocalAgreement:
    """Stabilizes rolling ASR hypotheses by committing words agreed across passes.

    The audio window slides, so each new hypothesis may start later than the last one.
    Each pass is therefore anchored on the tail of the confirmed text, and only the
    words after that anchor are compared with the previous pass.
    """

    def __init__(self, max_confirmed_words: int = MAX_CONFIRMED_WORDS):
        self.max_confirmed_words = max_confirmed_words
        self.pending_words: list[str] = []
        self.confirmed_words: list[str] = []

    @staticmethod
    def _normalize_token(tok: str) -> str:
        return tok.strip(".,!?;:\"'()[]{}\u0964\u2026").lower()

    def _unconfirmed(self, words: list[str]) -> list[str]:
        """Returns the part of `words` that comes after the already confirmed text."""
        tail = [self._normalize_token(w) for w in self.confirmed_words[-12:]]
        if not tail:
            return words
        norm = [self._normalize_token(w) for w in words]
        blocks = SequenceMatcher(None, tail, norm, autojunk=False).get_matching_blocks()
        for a, b, size in reversed(blocks):
            if size == 0:
                continue
            trailing = len(tail) - (a + size)  # confirmed words after this match
            if (size >= 2 and trailing <= 2) or (trailing == 0 and b <= 1):
                return words[b + size + trailing :]
        return words

    def update(self, transcript: str) -> tuple[str, str, list[str]]:
        words = transcript.split()
        if not words:
            return self.confirmed_text(), "", []

        fresh = self._unconfirmed(words)

        agreed = 0
        for a, b in zip(self.pending_words, fresh):
            if self._normalize_token(a) != self._normalize_token(b):
                break
            agreed += 1

        newly_agreed = fresh[:agreed]
        if newly_agreed:
            self.confirmed_words.extend(newly_agreed)
            self.confirmed_words = self.confirmed_words[-self.max_confirmed_words :]

        self.pending_words = fresh[agreed:]
        return self.confirmed_text(), " ".join(self.pending_words[-10:]), newly_agreed

    def confirmed_text(self) -> str:
        return " ".join(self.confirmed_words)

    def reset(self):
        self.pending_words.clear()
        self.confirmed_words.clear()


class CaptionTranslator:
    def __init__(self, mode: str, target_lang: str):
        self.mode = mode
        self.target_lang = target_lang
        self._cache: dict[str, str] = {}
        self._translator = None
        if mode == "target":
            if GoogleTranslator is None:
                logger.warning("deep-translator is not installed; target translation is disabled.")
            else:
                try:
                    self._translator = GoogleTranslator(source="auto", target=target_lang)
                    logger.info("Initialized GoogleTranslator for target language '%s'.", target_lang)
                except Exception as exc:
                    logger.error("Failed to initialize GoogleTranslator: %s", exc, exc_info=True)

    def translate_finalized(self, text: str) -> str:
        text = text.strip()
        if not text or self.mode in ("none", "to_en"):
            return ""
        if self.mode == "target":
            if text in self._cache:
                return self._cache[text]
            if self._translator is None:
                return "[Install deep-translator for target translation]"
            try:
                translated = self._translator.translate(text) or ""
                if translated:
                    self._cache[text] = translated
                    if len(self._cache) > 256:
                        self._cache.clear()
                return translated
            except Exception as exc:
                logger.error("Translation error for text '%s': %s", text[:60], exc, exc_info=True)
                return ""
        return ""


def find_wasapi_loopback(p: pa.PyAudio):
    wasapi_info = p.get_host_api_info_by_type(pa.paWASAPI)
    default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

    if default_speakers.get("isLoopbackDevice"):
        return default_speakers

    for loopback in p.get_loopback_device_info_generator():
        if default_speakers["name"] in loopback["name"]:
            return loopback

    raise RuntimeError(f"No WASAPI loopback device found for output '{default_speakers.get('name')}'.")


def audio_capture_worker(audio_q: queue.Queue, caption_q: queue.Queue, stop_event: threading.Event):
    p = None
    stream = None
    chunk_count = 0

    try:
        p = pa.PyAudio()
        device = find_wasapi_loopback(p)
        channels = max(1, int(device["maxInputChannels"]))
        native_rate = int(device["defaultSampleRate"])
        frames_per_buffer = max(1, int(native_rate * CHUNK_SECONDS))

        logger.info(
            "Opening WASAPI loopback device '%s' (index=%s, ch=%d, native_rate=%dHz -> %dHz).",
            device.get("name"),
            device.get("index"),
            channels,
            native_rate,
            SAMPLE_RATE,
        )

        stream = p.open(
            format=pa.paFloat32,
            channels=channels,
            rate=native_rate,
            input=True,
            input_device_index=device["index"],
            frames_per_buffer=frames_per_buffer,
        )
        logger.info("Audio loopback stream active. Waiting for system audio chunks...")

        while not stop_event.is_set():
            raw = stream.read(frames_per_buffer, exception_on_overflow=False)
            samples = np.frombuffer(raw, dtype=np.float32)

            if channels > 1:
                samples = samples.reshape(-1, channels).mean(axis=1)

            if native_rate != SAMPLE_RATE and len(samples) > 0:
                target_len = int(len(samples) * SAMPLE_RATE / native_rate)
                x_old = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
                x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
                samples = np.interp(x_new, x_old, samples).astype(np.float32)

            rms = float(np.sqrt(np.mean(np.square(samples)))) if len(samples) > 0 else 0.0
            chunk_count += 1
            if chunk_count == 1:
                logger.info(
                    "First audio chunk arrived (#1): %d samples (%.2fs) at %dHz, rms=%.4f.",
                    len(samples),
                    len(samples) / SAMPLE_RATE,
                    SAMPLE_RATE,
                    rms,
                )
            elif LOG_CHUNK_EVERY > 0 and chunk_count % LOG_CHUNK_EVERY == 0:
                logger.info(
                    "Audio chunk #%d arrived (%d samples, rms=%.4f, queue_size=%d/%d).",
                    chunk_count,
                    len(samples),
                    rms,
                    audio_q.qsize(),
                    audio_q.maxsize,
                )

            item = (samples, rms)
            try:
                audio_q.put_nowait(item)
            except queue.Full:
                logger.warning("audio_q full (%d items); dropping oldest chunk to keep latency low.", audio_q.maxsize)
                try:
                    audio_q.get_nowait()
                    audio_q.put_nowait(item)
                except queue.Empty:
                    pass

    except Exception as exc:
        logger.error("Audio capture error: %s", exc, exc_info=True)
        _publish_caption(caption_q, {"error_title": "Audio capture failed", "error_detail": str(exc)})
    finally:
        if stream is not None:
            try:
                if stream.is_active():
                    stream.stop_stream()
                stream.close()
            except Exception as exc:
                logger.error("Error while closing audio stream: %s", exc, exc_info=True)
        if p is not None:
            try:
                p.terminate()
            except Exception as exc:
                logger.error("Error while terminating PyAudio: %s", exc, exc_info=True)
        logger.info("Audio capture worker stopped (total chunks captured: %d).", chunk_count)


def _publish_caption(caption_q: queue.Queue, payload: dict):
    try:
        caption_q.put_nowait(payload)
    except queue.Full:
        logger.warning("caption_q full; replacing oldest caption frame.")
        try:
            caption_q.get_nowait()
            caption_q.put_nowait(payload)
        except queue.Empty:
            pass


def _async_translate(
    translator_obj: CaptionTranslator,
    phrase: str,
    caption_q: queue.Queue,
    lang_code: str,
    stop_event: threading.Event,
):
    if stop_event.is_set():
        return
    translated = translator_obj.translate_finalized(phrase)
    if translated and not stop_event.is_set():
        _publish_caption(caption_q, {"translation": translated, "lang": lang_code})


def _get_models_folder_stats(root_dir: Path) -> tuple[float, int]:
    """Returns (total_size_mb, file_count) across files in the project models directory."""
    total_bytes = 0
    file_count = 0
    try:
        if root_dir.exists():
            for entry in os.scandir(root_dir):
                if entry.is_file(follow_symlinks=False):
                    total_bytes += entry.stat(follow_symlinks=False).st_size
                    file_count += 1
                elif entry.is_dir(follow_symlinks=False):
                    for dirpath, _, filenames in os.walk(entry.path, followlinks=False):
                        for fn in filenames:
                            try:
                                total_bytes += os.path.getsize(os.path.join(dirpath, fn))
                                file_count += 1
                            except OSError:
                                pass
    except OSError:
        pass
    return total_bytes / (1024 * 1024), file_count


def asr_worker(
    audio_q: queue.Queue,
    caption_q: queue.Queue,
    stop_event: threading.Event,
    cfg: RuntimeConfig,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
):
    model_cache: dict[str, WhisperModel] = {}
    load_error = ""

    def load_model(size: str) -> WhisperModel | None:
        nonlocal load_error
        if size in model_cache:
            logger.info("Using cached Whisper model '%s' on '%s'.", size, device)
            return model_cache[size]

        if not CACHE_LOADED_MODELS and model_cache:
            model_cache.clear()

        logger.info(
            "Loading Whisper model '%s' into '%s' on device='%s' (compute_type='%s')...",
            size,
            MODELS_DIR,
            device,
            compute_type,
        )
        t0 = time.perf_counter()
        result_holder: dict[str, WhisperModel | BaseException | None] = {"model": None, "error": None}
        done_event = threading.Event()

        def _worker_load():
            try:
                result_holder["model"] = WhisperModel(
                    size,
                    device=device,
                    compute_type=compute_type,
                    download_root=str(MODELS_DIR),
                )
            except BaseException as exc:
                result_holder["error"] = exc
            finally:
                done_event.set()

        loader_t = threading.Thread(target=_worker_load, name="ModelLoader", daemon=True)
        loader_t.start()

        last_log_t = 0.0
        while not done_event.is_set() and not stop_event.is_set():
            elapsed = time.perf_counter() - t0
            size_mb, file_cnt = _get_models_folder_stats(MODELS_DIR)
            _publish_caption(
                caption_q,
                {
                    "loading_model": size,
                    "loading_meta": f"{size_mb:.1f} MB · {file_cnt} files · {elapsed:.1f} s elapsed",
                },
            )
            if (time.perf_counter() - last_log_t) >= 3.0:
                logger.info(
                    "Model '%s' download/load progress: %.1f MB across %d files (%.1fs).",
                    size,
                    size_mb,
                    file_cnt,
                    elapsed,
                )
                last_log_t = time.perf_counter()
            done_event.wait(timeout=0.35)

        if stop_event.is_set() and not done_event.is_set():
            return None

        err = result_holder["error"]
        loaded = result_holder["model"]
        if err is not None or not isinstance(loaded, WhisperModel):
            logger.error(
                "Failed to load Whisper model '%s' on device='%s': %s",
                size,
                device,
                err,
                exc_info=err,
            )
            load_error = str(err)
            return None

        size_mb, file_cnt = _get_models_folder_stats(MODELS_DIR)
        logger.info(
            "Whisper model '%s' loaded successfully in %.2fs on '%s' (models folder: %.1f MB, %d files).",
            size,
            time.perf_counter() - t0,
            device,
            size_mb,
            file_cnt,
        )
        model_cache[size] = loaded
        return loaded

    def drain_audio():
        while True:
            try:
                audio_q.get_nowait()
            except queue.Empty:
                return

    def fail(size: str):
        """Shows the load error on the overlay and idles until the user closes it."""
        if stop_event.is_set():
            return
        _publish_caption(
            caption_q,
            {"error_title": f"Could not load {size}", "error_detail": load_error or "See the console log."},
        )
        while not stop_event.wait(0.5):
            drain_audio()

    active_size, active_mode, active_src_lang, active_tgt_lang = cfg.snapshot()
    model = load_model(active_size)
    if model is None:
        fail(active_size)
        return
    drain_audio()  # audio captured while loading is stale
    _publish_caption(caption_q, {"loading_model": "", "loading_meta": ""})

    stabilizer = LocalAgreement(max_confirmed_words=MAX_CONFIRMED_WORDS)
    translator = CaptionTranslator(mode=active_mode, target_lang=active_tgt_lang)
    vad_params = {
        "threshold": VAD_THRESHOLD,
        "min_silence_duration_ms": VAD_MIN_SILENCE_MS,
    }

    max_samples = int(WINDOW_SECONDS * SAMPLE_RATE)
    rolling_audio = np.zeros(0, dtype=np.float32)
    last_infer_time = 0.0
    last_speech_time = time.monotonic()
    idle_cleared = True
    silent_chunks = 0
    stabilizer_cleared = False
    pending_phrase: list[str] = []
    last_lang = "" if active_src_lang == "auto" else active_src_lang
    chunks_ingested = 0
    pass_count = 0
    has_new_audio = False
    last_audio_time = time.monotonic()

    def flush_translation():
        """Translates words still waiting for a full phrase (target mode)."""
        if pending_phrase and active_mode == "target":
            threading.Thread(
                target=_async_translate,
                args=(translator, " ".join(pending_phrase), caption_q, last_lang, stop_event),
                daemon=True,
            ).start()
        pending_phrase.clear()

    def clear_idle_captions(now_ts: float, reason: str):
        nonlocal idle_cleared, rolling_audio
        if not idle_cleared and (now_ts - last_speech_time) >= IDLE_CLEAR_SECONDS:
            stabilizer.reset()
            idle_cleared = True
            if (now_ts - last_audio_time) >= IDLE_CLEAR_SECONDS:
                rolling_audio = np.zeros(0, dtype=np.float32)
            logger.info("Idle timeout (%.1fs) reached (%s); clearing overlay captions.", IDLE_CLEAR_SECONDS, reason)
            _publish_caption(
                caption_q,
                {
                    "confirmed": "",
                    "tentative": "",
                    "translation": "",
                    "active": False,
                    "rms": 0.0,
                    "lang": last_lang,
                },
            )
            flush_translation()  # after the clear, so the result is not wiped

    try:
        while not stop_event.is_set():
            cur_size, cur_mode, cur_src_lang, cur_tgt_lang = cfg.snapshot()
            if cfg.model_reload_event.is_set() or cur_size != active_size:
                cfg.model_reload_event.clear()
                requested_size = cur_size
                if requested_size != active_size:
                    _publish_caption(
                        caption_q,
                        {
                            "loading_model": requested_size,
                            "loading_meta": "Preparing model switch...",
                        },
                    )
                    if not CACHE_LOADED_MODELS:
                        # Release the old model first so both never sit in VRAM together.
                        model = None
                        model_cache.clear()
                        gc.collect()
                    model_next = load_model(requested_size)
                    if model_next is not None:
                        model = model_next
                        model_next = None  # keep a single reference, so it can be freed later
                        active_size = requested_size
                    elif not stop_event.is_set():
                        logger.warning("Switch to '%s' failed; returning to '%s'.", requested_size, active_size)
                        if cfg.snapshot()[0] == requested_size:
                            cfg.set_model_size(active_size)
                            cfg.model_reload_event.clear()
                        model = load_model(active_size)
                        if model is None:
                            fail(active_size)
                            return
                    stabilizer.reset()
                    pending_phrase.clear()
                    rolling_audio = np.zeros(0, dtype=np.float32)
                    has_new_audio = False
                    drain_audio()
                # Another switch may have been requested while this one was loading.
                if cfg.snapshot()[0] == active_size:
                    _publish_caption(caption_q, {"loading_model": "", "loading_meta": ""})
                continue

            if cur_src_lang != active_src_lang:
                active_src_lang = cur_src_lang
                stabilizer.reset()
                pending_phrase.clear()
                last_lang = "" if active_src_lang == "auto" else active_src_lang
                logger.info("Source language setting updated to '%s'.", active_src_lang)

            if (cur_mode, cur_tgt_lang) != (active_mode, active_tgt_lang):
                active_mode, active_tgt_lang = cur_mode, cur_tgt_lang
                translator = CaptionTranslator(mode=active_mode, target_lang=active_tgt_lang)
                stabilizer.reset()
                pending_phrase.clear()
                _publish_caption(caption_q, {"confirmed": "", "tentative": "", "translation": ""})
                logger.info("Active translation mode updated to '%s' (target='%s').", active_mode, active_tgt_lang)

            latest_rms = 0.0
            chunks = []
            try:
                chunks.append(audio_q.get(timeout=0.1))
                while True:
                    chunks.append(audio_q.get_nowait())
            except queue.Empty:
                pass

            for samples, rms in chunks:
                latest_rms = float(rms)
                chunks_ingested += 1
                has_new_audio = True
                last_audio_time = time.monotonic()
                rolling_audio = np.concatenate([rolling_audio, samples])
                if len(rolling_audio) > max_samples:
                    rolling_audio = rolling_audio[-max_samples:]

                if rms < SILENCE_RMS_THRESHOLD:
                    silent_chunks += 1
                else:
                    silent_chunks = 0
                    stabilizer_cleared = False

                if silent_chunks >= SILENCE_RESET_CHUNKS and not stabilizer_cleared:
                    stabilizer.reset()
                    rolling_audio = rolling_audio[-int(SAMPLE_RATE * 1.0) :]
                    stabilizer_cleared = True
                    idle_cleared = True
                    logger.info(
                        "Silence detected (%d low-RMS chunks < %.4f); cleared stale stabilizer state.",
                        silent_chunks,
                        SILENCE_RMS_THRESHOLD,
                    )
                    _publish_caption(
                        caption_q,
                        {
                            "confirmed": "",
                            "tentative": "",
                            "translation": "",
                            "active": False,
                            "rms": 0.0,
                            "lang": last_lang,
                        },
                    )
                    flush_translation()

            now = time.monotonic()
            clear_idle_captions(now, "no recent speech")

            if (
                has_new_audio
                and len(rolling_audio) >= int(SAMPLE_RATE * 0.75)
                and silent_chunks < SILENCE_RESET_CHUNKS
                and (now - last_infer_time) >= INFER_INTERVAL_SECONDS
            ):
                last_infer_time = now
                has_new_audio = False
                pass_count += 1
                task = "translate" if active_mode == "to_en" else "transcribe"
                lang_arg = None if active_src_lang == "auto" else active_src_lang
                audio_sec = len(rolling_audio) / SAMPLE_RATE

                infer_start = time.perf_counter()
                try:
                    segments, info = model.transcribe(
                        rolling_audio,
                        language=lang_arg,
                        task=task,
                        vad_filter=True,
                        vad_parameters=vad_params,
                        beam_size=1,
                        condition_on_previous_text=False,
                        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                    )
                    raw_text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
                    infer_ms = (time.perf_counter() - infer_start) * 1000.0
                    detected = getattr(info, "language", "") or ""
                    if lang_arg is not None:
                        last_lang = f"{lang_arg}*"
                    elif detected and audio_sec >= 1.0:
                        last_lang = detected
                except Exception as exc:
                    infer_ms = (time.perf_counter() - infer_start) * 1000.0
                    logger.error("ASR transcription error after %.1fms: %s", infer_ms, exc, exc_info=True)
                    continue

                if not raw_text:
                    logger.info(
                        "ASR pass #%d completed in %.1fms (buf=%.2fs, no text).",
                        pass_count,
                        infer_ms,
                        audio_sec,
                    )
                    _publish_caption(caption_q, {"active": False, "rms": latest_rms})
                    clear_idle_captions(now, "empty transcription")
                    continue

                last_speech_time = time.monotonic()  # after inference, which may be slow
                idle_cleared = False

                confirmed, tentative, newly_agreed = stabilizer.update(raw_text)
                logger.info(
                    "ASR pass #%d completed in %.1fms (buf=%.2fs, lang=%s, agreed=%d words).",
                    pass_count,
                    infer_ms,
                    audio_sec,
                    last_lang or "?",
                    len(newly_agreed),
                )

                if newly_agreed and active_mode == "target":
                    pending_phrase.extend(newly_agreed)
                    joined_phrase = " ".join(pending_phrase)
                    if len(pending_phrase) >= 6 or joined_phrase.endswith((".", "!", "?", "।")):
                        threading.Thread(
                            target=_async_translate,
                            args=(translator, joined_phrase, caption_q, last_lang, stop_event),
                            daemon=True,
                        ).start()
                        pending_phrase.clear()

                _publish_caption(
                    caption_q,
                    {
                        "confirmed": confirmed,
                        "tentative": tentative,
                        "active": True,
                        "rms": max(latest_rms, 0.05),
                        "lang": last_lang,
                    },
                )
    except Exception as exc:
        logger.error("Unexpected ASR worker error: %s", exc, exc_info=True)
    finally:
        model_cache.clear()
        logger.info(
            "ASR worker stopped (total audio chunks processed: %d, ASR passes: %d).",
            chunks_ingested,
            pass_count,
        )


def _resolve_font_family(candidates: list[str], fallback: str = "Arial") -> str:
    try:
        available = {f.lower(): f for f in tkfont.families()}
        for name in candidates:
            if name.lower() in available:
                return available[name.lower()]
    except Exception:
        pass
    return fallback


class FloatingCaptionOverlay:
    """High-DPI crisp floating video subtitle rendered on Canvas with subtle dark shadow:
    - Enables Windows DPI awareness so fonts render at native resolution without bitmap blur.
    - Renders floating subtitles and word segments on a `tk.Canvas` with a subtle dark shadow
      (`#0c0f14`, offset `+2px`) for readability over bright or dark video frames.
    - Keeps `-transparentcolor` (`#010203`) on backgrounds and hover-only top controls.
    """

    GWL_EXSTYLE = -20
    WS_EX_LAYERED = 0x00080000
    WS_EX_TRANSPARENT = 0x00000020

    # Transparent key color (never used for text or wave bars)
    BG_TRANSPARENT = "#010203"

    # Subtle dark shadow color (distinct from #010203 so it stays visible behind text)
    SHADOW_COLOR = "#0c0f14"

    # Crisp, high-contrast solid text & indicator colors
    TEXT_BRIGHT = "#ffffff"
    TEXT_TENTATIVE = "#4ef2c2"
    TEXT_MUTED = "#d6deeb"
    TEXT_TRANSLATION = "#ffe066"

    ACCENT_AMBER = "#ffc46b"
    ACCENT_BLUE = "#7fb0ff"
    ACCENT_BLUE_TEXT = "#cfe2ff"
    ACCENT_MINT = "#4ef2c2"
    ACCENT_ERROR = "#ff8f8f"

    def __init__(self, caption_q: queue.Queue, stop_event: threading.Event, cfg: RuntimeConfig):
        _enable_windows_dpi_awareness()

        self.caption_q = caption_q
        self.stop_event = stop_event
        self.cfg = cfg
        self.click_through = False
        self.last_lang = ""
        self.is_active_speech = False
        self.current_rms = 0.0
        self.confirmed_full = ""
        self.tentative_full = ""
        self.translation_text = ""
        self.translation_until = 0.0
        self.error_title = ""
        self.error_detail = ""
        self._hotkey_down = False
        self._last_caption_render = None
        self._mouse_over_overlay = False

        cur_size, cur_mode, cur_src_lang, _ = self.cfg.snapshot()
        self.loading_model = cur_size
        self.loading_meta = "Preparing model..."

        self._drag_x = 0
        self._drag_y = 0
        self._anim_tick = 0
        self._cursor_visible = True
        self._last_cursor_toggle = time.monotonic()
        self._last_loading_mode = None

        self.root = tk.Tk()
        self.root.title("Live Captions")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.configure(bg=self.BG_TRANSPARENT)
        try:
            self.root.attributes("-transparentcolor", self.BG_TRANSPARENT)
        except tk.TclError:
            self.root.attributes("-alpha", 0.92)

        self.font_caption = _resolve_font_family(
            ["Segoe UI", "Bahnschrift", "Arial", "Nirmala UI"]
        )
        self.font_body = _resolve_font_family(
            ["Segoe UI", "Arial", "Nirmala UI"]
        )
        self.font_mono = _resolve_font_family(
            ["Cascadia Mono", "Consolas", "Courier New"]
        )

        self.caption_font_obj = tkfont.Font(family=self.font_caption, size=25, weight="bold")
        self.listening_font_obj = tkfont.Font(family=self.font_caption, size=23, weight="bold")
        self.trans_font_obj = tkfont.Font(family=self.font_body, size=15, weight="bold")

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        self.win_w = min(1140, int(screen_w * 0.85))
        self.win_h = 175
        self.pos_x = (screen_w - self.win_w) // 2
        self.pos_y = int(screen_h * 0.77)
        self.root.geometry(f"{self.win_w}x{self.win_h}+{self.pos_x}+{self.pos_y}")

        self.outer_frame = tk.Frame(
            self.root,
            bg=self.BG_TRANSPARENT,
            highlightthickness=0,
            bd=0,
        )
        self.outer_frame.pack(fill=tk.BOTH, expand=True)

        # State 1: Loading view (transparent background, floating text & spinner)
        self.loading_frame = tk.Frame(self.outer_frame, bg=self.BG_TRANSPARENT)
        self.loading_canvas = tk.Canvas(
            self.loading_frame,
            bg=self.BG_TRANSPARENT,
            highlightthickness=0,
            bd=0,
        )
        self.loading_canvas.pack(fill=tk.BOTH, expand=True)

        # Top control strip (shown only when hovering over the overlay)
        self.toolbar_frame = tk.Frame(self.outer_frame, bg=self.BG_TRANSPARENT, height=36)

        self.model_btn = self._make_chip_button(
            self.toolbar_frame,
            label="Model",
            value=cur_size,
            keycap="1-3",
            command=self._on_cycle_model,
        )
        self.model_btn.pack(side=tk.LEFT, padx=(14, 10), pady=4)

        self.lang_btn = self._make_chip_button(
            self.toolbar_frame,
            label="Lang",
            value=self._format_lang_value(cur_src_lang),
            keycap="L",
            command=self._on_cycle_lang,
        )
        self.lang_btn.pack(side=tk.LEFT, padx=(0, 10), pady=4)

        self.mode_btn = self._make_chip_button(
            self.toolbar_frame,
            label="Mode",
            value=self._format_mode_value(cur_mode, cur_size),
            keycap="M",
            command=self._on_cycle_mode,
        )
        self.mode_btn.pack(side=tk.LEFT, padx=(0, 10), pady=4)

        self.badge_pill = tk.Label(
            self.toolbar_frame,
            text=self._format_badge_pill(cur_size, cur_src_lang),
            fg=self.TEXT_MUTED,
            bg=self.BG_TRANSPARENT,
            font=(self.font_mono, 10, "bold"),
            padx=8,
            pady=2,
        )
        self.badge_pill.pack(side=tk.RIGHT, padx=(6, 14), pady=4)

        self.esc_cap = tk.Label(
            self.toolbar_frame,
            text="[Esc]",
            fg=self.TEXT_BRIGHT,
            bg=self.BG_TRANSPARENT,
            font=(self.font_mono, 10, "bold"),
            padx=4,
            pady=2,
        )
        self.esc_cap.pack(side=tk.RIGHT, padx=2, pady=4)

        self.click_status_lbl = tk.Label(
            self.toolbar_frame,
            text="Click-through off",
            fg=self.TEXT_MUTED,
            bg=self.BG_TRANSPARENT,
            font=(self.font_body, 11, "bold"),
            padx=6,
        )
        self.click_status_lbl.pack(side=tk.RIGHT, padx=(0, 4), pady=4)

        # Main caption area (wave bars + Canvas floating subtitle renderer)
        self.body_frame = tk.Frame(self.outer_frame, bg=self.BG_TRANSPARENT)

        self.wave_canvas = tk.Canvas(
            self.body_frame,
            bg=self.BG_TRANSPARENT,
            highlightthickness=0,
            bd=0,
            width=64,
            height=68,
        )
        self.wave_canvas.pack(side=tk.LEFT, padx=(22, 14), pady=10)

        self.caption_canvas = tk.Canvas(
            self.body_frame,
            bg=self.BG_TRANSPARENT,
            highlightthickness=0,
            bd=0,
            height=112,
        )
        self.caption_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 22), pady=8)

        # A root binding already covers every child widget; binding both toggles twice.
        self.root.bind("<Button-3>", self._toggle_click_through)
        self.root.bind("<ButtonPress-1>", self._start_drag)
        self.root.bind("<B1-Motion>", self._do_drag)

        # Hotkeys (when overlay has focus)
        self.root.bind("<Key-l>", self._on_cycle_lang)
        self.root.bind("<Key-L>", self._on_cycle_lang)
        self.root.bind("<Key-m>", self._on_cycle_mode)
        self.root.bind("<Key-M>", self._on_cycle_mode)
        self.root.bind("<Key-1>", lambda _e: self._on_select_model("small"))
        self.root.bind("<Key-2>", lambda _e: self._on_select_model("medium"))
        self.root.bind("<Key-3>", lambda _e: self._on_select_model("large-v3-turbo"))
        self.root.bind("<Escape>", lambda _e: self._on_close())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._sync_view_mode(force=True)
        self.root.after(200, self.root.focus_force)
        self._poll_ui()

    @staticmethod
    def _format_lang_value(src_lang: str) -> str:
        return "Auto" if src_lang == "auto" else src_lang.upper()

    @staticmethod
    def _format_mode_value(mode: str, model_size: str = "") -> str:
        if mode == "to_en" and "turbo" in model_size:
            return "To EN: press 1 or 2"  # turbo cannot translate
        mapping = {"none": "None", "to_en": "To EN", "target": "Target"}
        return mapping.get(mode, mode)

    @staticmethod
    def _format_badge_pill(model_size: str, src_lang: str) -> str:
        short_model = "turbo" if "turbo" in model_size else model_size
        lang_tag = "AUTO" if src_lang == "auto" else src_lang.upper()
        return f"{lang_tag} · {short_model}"

    def _make_chip_button(self, parent, label: str, value: str, keycap: str, command):
        chip = tk.Frame(
            parent,
            bg=self.BG_TRANSPARENT,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        lbl = tk.Label(
            chip,
            text=f"{label}:",
            fg=self.TEXT_MUTED,
            bg=self.BG_TRANSPARENT,
            font=(self.font_body, 11, "bold"),
            padx=2,
        )
        lbl.pack(side=tk.LEFT, padx=(4, 2), pady=3)

        val = tk.Label(
            chip,
            text=value,
            fg=self.TEXT_BRIGHT,
            bg=self.BG_TRANSPARENT,
            font=(self.font_body, 11, "bold"),
            padx=2,
        )
        val.pack(side=tk.LEFT, padx=(0, 4), pady=3)

        cap = tk.Label(
            chip,
            text=f"[{keycap}]",
            fg=self.ACCENT_BLUE_TEXT,
            bg=self.BG_TRANSPARENT,
            font=(self.font_mono, 10, "bold"),
            padx=2,
            pady=3,
        )
        cap.pack(side=tk.LEFT, padx=(0, 4), pady=3)

        chip._val_label = val
        for w in (chip, lbl, val, cap):
            w.bind("<Button-1>", lambda _e, cmd=command: cmd())
        return chip

    def _refresh_controls(self):
        cur_size, cur_mode, cur_src_lang, _ = self.cfg.snapshot()
        self.model_btn._val_label.configure(text=cur_size)
        self.lang_btn._val_label.configure(text=self._format_lang_value(cur_src_lang))
        self.mode_btn._val_label.configure(text=self._format_mode_value(cur_mode, cur_size))
        self.click_status_lbl.configure(
            text="Click-through on · Ctrl+Alt+C" if self.click_through else "Click-through off"
        )
        self.badge_pill.configure(text=self._format_badge_pill(cur_size, cur_src_lang))

    def _has_caption_content(self) -> bool:
        return bool(self.confirmed_full.strip() or self.tentative_full.strip())

    def _is_mouse_over_overlay(self) -> bool:
        if self.click_through:
            return False
        try:
            mx = self.root.winfo_pointerx()
            my = self.root.winfo_pointery()
            x0 = self.root.winfo_rootx()
            y0 = self.root.winfo_rooty()
            x1 = x0 + self.root.winfo_width()
            y1 = y0 + self.root.winfo_height()
            return x0 <= mx <= x1 and y0 <= my <= y1
        except tk.TclError:
            return False

    def _sync_view_mode(self, force: bool = False):
        is_loading = bool(self.loading_model or self.error_title)
        if force or is_loading != self._last_loading_mode:
            self._last_loading_mode = is_loading
            if is_loading:
                self.toolbar_frame.pack_forget()
                self.body_frame.pack_forget()
                self.loading_frame.pack(fill=tk.BOTH, expand=True)
            else:
                self.loading_frame.pack_forget()
                self.body_frame.pack(fill=tk.BOTH, expand=True)

        if not is_loading:
            if self._mouse_over_overlay:
                if not self.toolbar_frame.winfo_manager():
                    self.toolbar_frame.pack(in_=self.outer_frame, before=self.body_frame, fill=tk.X)
            else:
                self.toolbar_frame.pack_forget()

    def _draw_shadow_text(
        self,
        canvas: tk.Canvas,
        x: int,
        y: int,
        text: str,
        fill: str,
        font,
        anchor: str = "w",
        width: int | None = None,
        shadow_offset: int = 2,
    ):
        """Draws text with a subtle dark shadow offset for crisp contrast over any video."""
        if not text:
            return
        kw = {"text": text, "font": font, "anchor": anchor}
        if width is not None:
            kw["width"] = width

        canvas.create_text(x + shadow_offset, y + shadow_offset, fill=self.SHADOW_COLOR, **kw)
        canvas.create_text(x, y, fill=fill, **kw)

    def _draw_loading_card(self):
        c = self.loading_canvas
        w = max(400, c.winfo_width() or self.win_w)
        h = max(90, c.winfo_height() or self.win_h)
        c.delete("all")

        if self.error_title:
            self._draw_shadow_text(
                c, 28, 40, self.error_title, self.ACCENT_ERROR, (self.font_caption, 19, "bold"), anchor="w"
            )
            detail = " ".join(self.error_detail.split())
            if len(detail) > 220:
                detail = detail[:217] + "..."
            self._draw_shadow_text(
                c, 28, 68, detail, self.TEXT_BRIGHT, (self.font_mono, 11, "bold"), anchor="nw", width=w - 56
            )
            self._draw_shadow_text(
                c, 28, h - 24, "Click here, then press Esc to close", self.TEXT_MUTED, (self.font_body, 11, "bold"), anchor="w"
            )
            return

        cx, cy, r = 56, h // 2 + 2, 24
        c.create_oval(cx - r + 2, cy - r + 2, cx + r + 2, cy + r + 2, outline=self.SHADOW_COLOR, width=4)
        c.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#8c6d3f", width=4)
        start_deg = (self._anim_tick * 16) % 360
        c.create_arc(
            cx - r,
            cy - r,
            cx + r,
            cy + r,
            start=start_deg,
            extent=95,
            style=tk.ARC,
            outline=self.ACCENT_AMBER,
            width=5,
        )

        title_x = cx + r + 24
        self._draw_shadow_text(
            c, title_x, cy - 12, f"Loading {self.loading_model}", self.TEXT_BRIGHT, (self.font_caption, 20, "bold"), anchor="w"
        )
        self._draw_shadow_text(
            c, title_x, cy + 14, self.loading_meta, self.TEXT_MUTED, (self.font_mono, 11, "bold"), anchor="w"
        )
        self._draw_shadow_text(
            c, w - 24, cy, "Captions start automatically", self.TEXT_MUTED, (self.font_body, 11, "bold"), anchor="e"
        )

    def _draw_wave_bars(self):
        c = self.wave_canvas
        c.delete("all")
        w = int(c["width"])
        h = int(c["height"])
        cy = h // 2

        bar_w = 6
        gap = 5
        total_w = 5 * bar_w + 4 * gap
        start_x = (w - total_w) // 2

        if self.is_active_speech:
            color = self.ACCENT_MINT
            base_heights = [20, 38, 48, 32, 20]
            amp_boost = min(12.0, self.current_rms * 90.0)
            for idx, base_h in enumerate(base_heights):
                phase = self._anim_tick * 0.45 + idx * 1.15
                factor = 0.62 + 0.38 * abs(np.sin(phase))
                bh = max(12, min(h - 6, int(base_h * factor + amp_boost * 0.25)))
                x0 = start_x + idx * (bar_w + gap)
                y0, y1 = cy - bh // 2, cy + bh // 2
                c.create_rectangle(x0 + 2, y0 + 2, x0 + bar_w + 2, y1 + 2, fill=self.SHADOW_COLOR, outline="")
                c.create_rectangle(x0, y0, x0 + bar_w, y1, fill=color, outline="")
        else:
            color = self.ACCENT_BLUE
            idle_heights = [10, 16, 22, 16, 10]
            for idx, base_h in enumerate(idle_heights):
                phase = self._anim_tick * 0.14 + idx * 0.75
                factor = 0.75 + 0.25 * abs(np.sin(phase))
                bh = max(8, int(base_h * factor))
                x0 = start_x + idx * (bar_w + gap)
                y0, y1 = cy - bh // 2, cy + bh // 2
                c.create_rectangle(x0 + 2, y0 + 2, x0 + bar_w + 2, y1 + 2, fill=self.SHADOW_COLOR, outline="")
                c.create_rectangle(x0, y0, x0 + bar_w, y1, fill=color, outline="")

    def _render_caption_lines(self):
        conf_words = self.confirmed_full.split()
        tent_words = self.tentative_full.split()[-5:]
        combined = [("bright", w) for w in conf_words] + [("tentative", w) for w in tent_words]
        window_items = combined[-MAX_SUBTITLE_WORDS:]

        _, cur_mode, _, _ = self.cfg.snapshot()
        active_trans = self.translation_text.strip() if cur_mode != "none" else ""
        show_captions = self.is_active_speech or bool(window_items) or bool(active_trans)

        state = (
            show_captions,
            tuple(window_items),
            active_trans,
            self._cursor_visible,
            self.caption_canvas.winfo_width(),
        )
        if state == self._last_caption_render:
            return
        self._last_caption_render = state

        c = self.caption_canvas
        c.delete("all")
        avail_w = max(240, (c.winfo_width() or (self.win_w - 130)) - 12)

        if not show_captions:
            self._draw_shadow_text(
                c,
                6,
                34,
                "Listening for system audio…",
                self.ACCENT_BLUE_TEXT,
                self.listening_font_obj,
                anchor="w",
                width=avail_w,
                shadow_offset=2,
            )
            return

        # Wrap tokens onto lines cleanly so each line has its own sharp shadow
        tokens: list[tuple[str, str]] = [(tag, w) for tag, w in window_items]
        if (window_items or self.is_active_speech) and self._cursor_visible:
            tokens.append(("cursor_on", "│"))

        lines: list[list[tuple[str, str]]] = []
        cur_line: list[tuple[str, str]] = []
        cur_line_w = 0
        space_w = self.caption_font_obj.measure(" ")

        for tag, word in tokens:
            word_w = self.caption_font_obj.measure(word)
            needed_w = word_w if not cur_line else (cur_line_w + space_w + word_w)
            if cur_line and needed_w > avail_w:
                lines.append(cur_line)
                cur_line = [(tag, word)]
                cur_line_w = word_w
            else:
                cur_line.append((tag, word))
                cur_line_w = needed_w
        if cur_line:
            lines.append(cur_line)

        lines = lines[-2:]  # Keep subtitle compact (up to 2 lines)

        color_map = {
            "bright": self.TEXT_BRIGHT,
            "tentative": self.TEXT_TENTATIVE,
            "cursor_on": self.ACCENT_MINT,
        }

        line_h = self.caption_font_obj.metrics("linespace") + 4
        start_y = 10

        for line_idx, line_tokens in enumerate(lines):
            y = start_y + line_idx * line_h
            x = 6
            for tok_idx, (tag, word) in enumerate(line_tokens):
                if tok_idx > 0:
                    x += space_w
                fill = color_map.get(tag, self.TEXT_BRIGHT)
                self._draw_shadow_text(c, x, y, word, fill, self.caption_font_obj, anchor="nw", shadow_offset=2)
                x += self.caption_font_obj.measure(word)

        if active_trans:
            trans_y = start_y + len(lines) * line_h + 4
            self._draw_shadow_text(
                c,
                6,
                trans_y,
                active_trans,
                self.TEXT_TRANSLATION,
                self.trans_font_obj,
                anchor="nw",
                width=avail_w,
                shadow_offset=2,
            )

    def _on_cycle_lang(self, _event=None):
        new_lang = self.cfg.cycle_source_lang()
        self.last_lang = "" if new_lang == "auto" else f"{new_lang}*"
        logger.info("Switched source language setting to '%s'.", new_lang)
        self._refresh_controls()

    def _on_cycle_mode(self, _event=None):
        new_mode = self.cfg.cycle_translate_mode()
        if new_mode == "none":
            self.translation_text = ""
        logger.info("Switched TRANSLATE_MODE to '%s'.", new_mode)
        self._refresh_controls()
        self._sync_view_mode(force=True)

    def _on_select_model(self, size: str):
        if size == self.cfg.snapshot()[0]:
            return  # already selected: nothing to load
        self.cfg.set_model_size(size)
        self.loading_model = size
        self.loading_meta = "Switching model..."
        logger.info("Requested MODEL_SIZE switch to '%s'.", size)
        self._refresh_controls()
        self._sync_view_mode(force=True)

    def _on_cycle_model(self):
        new_size = self.cfg.cycle_model_size()
        self.loading_model = new_size
        self.loading_meta = "Switching model..."
        logger.info("Cycled MODEL_SIZE to '%s'.", new_size)
        self._refresh_controls()
        self._sync_view_mode(force=True)

    def _set_win32_click_through(self, enabled: bool):
        try:
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id()) or self.root.winfo_id()
            ex_style = ctypes.windll.user32.GetWindowLongW(hwnd, self.GWL_EXSTYLE)
            if enabled:
                ex_style |= self.WS_EX_LAYERED | self.WS_EX_TRANSPARENT
            else:
                ex_style &= ~self.WS_EX_TRANSPARENT
            ctypes.windll.user32.SetWindowLongW(hwnd, self.GWL_EXSTYLE, ex_style)
        except Exception as exc:
            logger.error("Failed to toggle Win32 click-through style: %s", exc, exc_info=True)

    def _start_drag(self, event):
        self.root.focus_force()  # a borderless window needs this for hotkeys
        self._drag_x = event.x_root - self.root.winfo_x()
        self._drag_y = event.y_root - self.root.winfo_y()

    def _do_drag(self, event):
        self.pos_x = event.x_root - self._drag_x
        self.pos_y = event.y_root - self._drag_y
        self.root.geometry(f"+{self.pos_x}+{self.pos_y}")

    def _poll_unlock_hotkey(self):
        """Global Ctrl+Alt+C. A click-through window cannot receive clicks to unlock itself."""
        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return
        pressed = all(windll.user32.GetAsyncKeyState(vk) & 0x8000 for vk in (0x11, 0x12, 0x43))
        if pressed and not self._hotkey_down:
            self._toggle_click_through()
        self._hotkey_down = pressed

    def _toggle_click_through(self, _event=None):
        self.click_through = not self.click_through
        self._set_win32_click_through(self.click_through)
        mode_tag = "CLICK-THROUGH ON" if self.click_through else "INTERACTIVE"
        logger.info("Overlay click-through toggled: %s", mode_tag)
        self._refresh_controls()

    def _apply_payload(self, payload: dict):
        if "error_title" in payload:
            self.error_title = payload["error_title"]
            self.error_detail = payload.get("error_detail", "")
        if "loading_model" in payload:
            self.loading_model = payload["loading_model"]
        if "loading_meta" in payload:
            self.loading_meta = payload["loading_meta"]
        if "confirmed" in payload:
            self.confirmed_full = payload["confirmed"]
        if "tentative" in payload:
            self.tentative_full = payload["tentative"]
        if "translation" in payload:
            self.translation_text = payload["translation"]
            if self.translation_text:
                self.translation_until = time.monotonic() + TRANSLATION_HOLD_SECONDS
        if "active" in payload:
            self.is_active_speech = bool(payload["active"])
        if "rms" in payload:
            self.current_rms = float(payload["rms"])
        if payload.get("lang"):
            self.last_lang = payload["lang"]

    def _poll_ui(self):
        try:
            if self.stop_event.is_set():
                logger.info("Stop signal received; closing overlay window.")
                self.root.destroy()
                return

            while True:
                try:
                    self._apply_payload(self.caption_q.get_nowait())
                except queue.Empty:
                    break
            self._poll_unlock_hotkey()

            if (
                self.translation_text
                and not self._has_caption_content()
                and time.monotonic() >= self.translation_until
            ):
                self.translation_text = ""

            now = time.monotonic()
            if (now - self._last_cursor_toggle) >= 0.55:
                self._cursor_visible = not self._cursor_visible
                self._last_cursor_toggle = now

            self._anim_tick += 1
            self._mouse_over_overlay = self._is_mouse_over_overlay()
            self._refresh_controls()
            self._sync_view_mode()

            if self.loading_model or self.error_title:
                self._draw_loading_card()
            else:
                self._draw_wave_bars()
                self._render_caption_lines()
        except Exception as exc:
            logger.error("Overlay UI polling error: %s", exc, exc_info=True)
        finally:
            if not self.stop_event.is_set():
                self.root.after(45, self._poll_ui)

    def _on_close(self):
        logger.info("Overlay close requested by user.")
        self.stop_event.set()
        try:
            self.root.destroy()
        except Exception as exc:
            logger.error("Error destroying overlay window: %s", exc, exc_info=True)

    def run(self):
        logger.info("Starting FloatingCaptionOverlay UI loop.")
        try:
            self.root.mainloop()
        except Exception as exc:
            logger.error("Overlay mainloop error: %s", exc, exc_info=True)
            self.stop_event.set()


def main():
    logger.info("Initializing Live Caption Overlay pipeline (models dir: %s)...", MODELS_DIR)
    cfg = RuntimeConfig(
        model_size=DEFAULT_MODEL_SIZE,
        translate_mode=DEFAULT_TRANSLATE_MODE,
        source_lang=DEFAULT_SOURCE_LANG,
        target_lang=TARGET_LANG,
    )
    audio_q: queue.Queue = queue.Queue(maxsize=30)
    caption_q: queue.Queue = queue.Queue(maxsize=30)
    stop_event = threading.Event()

    t_audio = threading.Thread(
        target=audio_capture_worker,
        name="AudioCapture",
        args=(audio_q, caption_q, stop_event),
        daemon=True,
    )
    t_asr = threading.Thread(
        target=asr_worker,
        name="ASRWorker",
        args=(audio_q, caption_q, stop_event, cfg),
        daemon=True,
    )

    t_audio.start()
    t_asr.start()

    try:
        overlay = FloatingCaptionOverlay(caption_q, stop_event, cfg)
        overlay.run()
    except Exception as exc:
        logger.error("Fatal error in main pipeline: %s", exc, exc_info=True)
    finally:
        stop_event.set()
        t_audio.join(timeout=2.0)
        t_asr.join(timeout=2.0)
        logger.info("Live Caption Overlay pipeline shut down cleanly.")


if __name__ == "__main__":
    main()