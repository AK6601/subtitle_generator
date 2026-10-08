#!/usr/bin/env python3
"""
Local subtitle generator and translator.

Whisper transcribes the video, Ollama translates the transcript.

The translator sends batches of subtitles together with a rolling window of
already-translated context, and constrains Ollama to a JSON schema, so the
model cannot leak reasoning, labels or source-language text into the SRT.
"""

import argparse
import ctypes
import gc
import json
import re
import sys
from pathlib import Path

import ollama


# ============================================================
# CONFIGURATION
# ============================================================

# Whisper model (run through faster-whisper / CTranslate2):
# "large"  = large-v3, best quality, very slow on CPU
# "medium" = good compromise
# "small"  = much faster
WHISPER_MODEL = "large"

# None = use CUDA when available, otherwise CPU
WHISPER_DEVICE = None

# Numeric precision for Whisper.
# "float32"      = full precision, no quality loss. Fast on any NVIDIA GPU.
# "int8_float32" = faster and smaller, very slightly less accurate.
# "float16"      = only worth it on Volta (RTX 20xx) or newer; Pascal cards
#                  (GTX 10xx) run fp16 at a fraction of their fp32 speed.
WHISPER_COMPUTE_TYPE = "float32"

# Skip silence with a voice-activity detector before transcribing. Faster,
# and removes most hallucinated lines, but changes how lines are split.
WHISPER_VAD_FILTER = False

# Optional style hint for Whisper. Keep it short, or leave it None:
# a long prompt makes Whisper hallucinate.
WHISPER_INITIAL_PROMPT = None

# Ollama model. Needs to be uncensored for explicit material,
# otherwise the model refuses and lines fall back to the source text.
LLM_MODEL = "richardyoung/qwen3-14b-abliterated"

# Source / target language
SOURCE_LANGUAGE = "ja"
SOURCE_LANGUAGE_NAME = "Japanese"
TARGET_LANGUAGE_NAME = "English"

# One sentence about the video. This is the single most effective knob for
# translation quality: Japanese drops subjects constantly, and the model
# needs to know who is speaking to whom.
CONTENT_HINT = (
    "You are an expert literary and entertainment translation agent specializing in Japanese adult media. Your goal is to translate source text accurately while preserving the specific emotional tone, stylistic nuances, character dynamics, and stylistic tropes unique to Japanese adult fiction, adapting them naturally into the target language."
)

# How many subtitles are translated in one request.
# Bigger = more context and faster, but more room for the model to drift.
BATCH_SIZE = 8

# How many already-translated lines are shown as context.
CONTEXT_LINES = 4

# Retries for a batch before falling back to line-by-line
MAX_TRANSLATION_RETRIES = 3

# Temperature per attempt. Attempt 1 is deterministic; later attempts get
# some randomness, otherwise a retry just reproduces the same bad answer.
TEMPERATURE_LADDER = (0.0, 0.3, 0.6)

# Context window for the LLM. Too small silently truncates the prompt.
LLM_NUM_CTX = 8192

# Keep the model resident between batches
LLM_KEEP_ALIVE = "10m"

# Reject a translation longer than this (characters)
MAX_SUBTITLE_CHARS = 220

# Line wrapping in the output SRT
SUBTITLE_LINE_WIDTH = 42
MAX_SUBTITLE_LINES = 2

# Merge consecutive segments with identical text into one entry.
# Fixes Whisper repetition loops.
COLLAPSE_REPEATS = True

# Never leave a subtitle on screen longer than this (seconds). Whisper likes
# to stretch one hallucinated line over a long silence.
MAX_SUBTITLE_DURATION = 8.0

# Only merge repeats this close together (seconds). Without this, the same
# word said twice a minute apart becomes one subtitle hanging on screen for
# the whole gap.
MAX_MERGE_GAP = 1.5

# Drop segments that are only non-verbal sounds (ああ, うっ, んっ ...).
# Set to False to keep them.
DROP_VOCALIZATIONS = True

# Drop segments Whisper itself flagged as probably-silence
NO_SPEECH_THRESHOLD = 0.8

# True = always run Whisper even if a source .srt exists
FORCE_WHISPER = False

# True = delete and recreate the translated SRT
FORCE_TRANSLATE = False

# Default video when none is passed on the command line
VIDEO_FILE = ""


# ============================================================
# TIMESTAMPS
# ============================================================

def format_timestamp(seconds):
    """
    Convert seconds to an SRT timestamp.

    Example:
        3.48 -> 00:00:03,480
    """

    seconds = max(0.0, float(seconds))

    total_seconds = int(seconds)
    milliseconds = int(round((seconds - total_seconds) * 1000))

    if milliseconds >= 1000:
        total_seconds += 1
        milliseconds -= 1000

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def parse_timestamp(text):
    """
    Parse an SRT timestamp into seconds.

    Accepts both "00:00:03,480" and "00:00:03.480".
    """

    match = re.match(
        r"\s*(\d+):(\d{2}):(\d{2})[,.](\d{1,3})\s*",
        str(text)
    )

    if not match:
        return 0.0

    hours, minutes, secs, millis = match.groups()

    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(secs)
        + int(millis.ljust(3, "0")) / 1000.0
    )


# ============================================================
# TEXT HELPERS
# ============================================================

CJK_PATTERN = re.compile(
    r"[぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ]"
)


CJK_RUN_PATTERN = re.compile(
    r"[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff"
    r"\uf900-\ufaff\uff01-\uff60\uff65-\uff9f]+"
)


def normalize_text(text):
    """Line endings, control characters and stray whitespace."""

    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")

    text = re.sub(r"[​-‏‪-‮﻿]", "", text)

    return text.strip()


def has_source_script(text):
    """True if the text still contains Japanese characters."""

    return bool(CJK_PATTERN.search(text or ""))


def strip_source_script(text):
    """
    Remove source-language fragments from a translation.

    Some models prefix the answer with their own instructions echoed back in
    the source language ("段落の区切りを保つ"). The English half of such an
    answer is usually fine, so salvage it instead of throwing the line away.
    """

    if not has_source_script(text):
        return text

    lines = text.splitlines()
    clean = [line for line in lines if not has_source_script(line)]

    if clean:
        text = "\n".join(clean)

    # Fragments sitting inline with the translation
    if has_source_script(text):
        text = CJK_RUN_PATTERN.sub(" ", text)

    # A list number left behind by the removed fragment
    text = re.sub(r"^\s*\d+\s*[.)\-:]?\s+", "", text)

    return re.sub(r"\s{2,}", " ", text).strip()


# Non-verbal sounds, after punctuation and elongation marks are stripped.
# Written out explicitly so that real words ("はい", "いや") are not caught.
VOCALIZATION_PATTERNS = tuple(
    re.compile(pattern) for pattern in (
        r"^[あぁ]+$",
        r"^[いぃ]+$",
        r"^[うぅ]+$",
        r"^[えぇ]+$",
        r"^[おぉ]+$",
        r"^[んッっ]+$",
        r"^[あぁ][んッっ]+$",
        r"^[うぅ][んッっ]+$",
        r"^[んン][あぁ]+$",
        r"^[はひふへほ][ぁあ]*$",
        r"^[やゃ][ぁあ]+$",
        r"^[ふフ][うぅ]+$",
        r"^[アァ]+$",
        r"^[ウゥ]+$",
    )
)


def is_vocalization(text):
    """
    Detect moans / gasps / breathing, which carry no translatable content
    and which Whisper loves to emit dozens of times in a row.
    """

    stripped = re.sub(r"[\s。、，,.!?！？…ー〜~\-–—「」『』()（）]", "", text or "")

    if not stripped:
        return True

    return any(
        pattern.match(stripped)
        for pattern in VOCALIZATION_PATTERNS
    )


JUNK_EXACT = {
    "字幕",
    "字幕翻訳",
    "字幕翻译",
    "字幕翻訳機",
    "字幕翻译机",
    "字幕翻譯",
    "翻訳",
    "翻译",
    "翻譯",
    "subtitle",
    "subtitles",
    "subtitle translator",
    "translator",
}


def is_junk_text(text):
    """Whisper hallucinations, watermarks and formatting artifacts."""

    if not text:
        return True

    cleaned = text.strip()

    if len(cleaned) < 2:
        return True

    if cleaned.lower() in {item.lower() for item in JUNK_EXACT}:
        return True

    # Pure numbers / punctuation
    if re.fullmatch(r"[\d\s.,:;+\-=/\\|]+", cleaned):
        return True

    if "-->" in cleaned:
        return True

    # Alignment tags from burned-in subtitle tracks
    if re.fullmatch(r"(?i)\s*\.?(align|center|left|right)\s*", cleaned):
        return True

    # A single symbol that is not actually Japanese
    if len(cleaned) <= 2 and not has_source_script(cleaned):
        return True

    return False


def wrap_subtitle(text, width=SUBTITLE_LINE_WIDTH, max_lines=MAX_SUBTITLE_LINES):
    """Greedy wrap so a subtitle stays readable on screen."""

    words = text.split()

    if not words:
        return ""

    lines = []
    current = ""

    for word in words:

        candidate = f"{current} {word}".strip()

        if len(candidate) <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word

    if current:
        lines.append(current)

    # Never exceed max_lines: push the overflow into the last line
    if len(lines) > max_lines:
        head = lines[:max_lines - 1]
        tail = " ".join(lines[max_lines - 1:])
        lines = head + [tail]

    return "\n".join(lines)


# ============================================================
# ENTRY CLEANUP
# ============================================================

def clean_entries(entries):
    """
    Turn raw segments into a usable subtitle list.

    Drops junk, drops non-verbal sounds, merges repetition loops and
    renumbers the result. Applied to Whisper output and to an existing
    source SRT alike, so a re-run also cleans up an old bad transcript.
    """

    kept = []
    dropped_junk = 0
    dropped_sound = 0
    merged = 0

    for entry in entries:

        text = normalize_text(entry.get("text", ""))

        if is_junk_text(text):
            dropped_junk += 1
            continue

        if DROP_VOCALIZATIONS and is_vocalization(text):
            dropped_sound += 1
            continue

        if entry.get("no_speech_prob", 0.0) > NO_SPEECH_THRESHOLD:
            dropped_junk += 1
            continue

        # Merge a repeated line into the previous entry
        if (
            COLLAPSE_REPEATS
            and kept
            and kept[-1]["text"] == text
            and float(entry["start"]) - kept[-1]["end"] <= MAX_MERGE_GAP
        ):
            kept[-1]["end"] = max(kept[-1]["end"], entry["end"])
            merged += 1
            continue

        kept.append({
            "start": float(entry["start"]),
            "end": float(entry["end"]),
            "text": text,
        })

    for position, entry in enumerate(kept, start=1):

        entry["index"] = position

        # Guard against zero-length or inverted timings
        if entry["end"] <= entry["start"]:
            entry["end"] = entry["start"] + 1.0

        # Cap how long a single subtitle stays on screen
        entry["end"] = min(
            entry["end"],
            entry["start"] + MAX_SUBTITLE_DURATION,
        )

    if dropped_junk or dropped_sound or merged:

        print(
            f"\nCleanup: merged {merged} repeated, "
            f"dropped {dropped_sound} non-verbal, "
            f"dropped {dropped_junk} junk."
        )

    return kept


# ============================================================
# SRT IO
# ============================================================

def parse_srt(srt_path):
    """Parse an SRT file into entries with float timings."""

    with open(srt_path, "r", encoding="utf-8-sig") as handle:
        content = handle.read()

    entries = []

    for block in re.split(r"\n\s*\n", content.strip()):

        lines = [line for line in block.splitlines() if line.strip()]

        if len(lines) < 2:
            continue

        # An index line is optional in the wild
        offset = 0

        if re.fullmatch(r"\s*\d+\s*", lines[0]):
            offset = 1

        if len(lines) <= offset:
            continue

        timing = re.match(r"(.+?)\s*-->\s*(.+)", lines[offset])

        if not timing:
            continue

        text = "\n".join(lines[offset + 1:]).strip()

        if not text:
            continue

        entries.append({
            "index": len(entries) + 1,
            "start": parse_timestamp(timing.group(1)),
            "end": parse_timestamp(timing.group(2)),
            "text": text,
        })

    return entries


def write_srt_entry(handle, index, start, end, text):
    """Write one SRT entry and flush, so progress survives a crash."""

    handle.write(
        f"{index}\n"
        f"{format_timestamp(start)} --> {format_timestamp(end)}\n"
        f"{text}\n\n"
    )

    handle.flush()


def write_srt(path, entries):
    """Write a complete SRT file."""

    with open(path, "w", encoding="utf-8") as handle:

        for position, entry in enumerate(entries, start=1):

            write_srt_entry(
                handle,
                position,
                entry["start"],
                entry["end"],
                entry["text"],
            )


# ============================================================
# WHISPER
# ============================================================

def resolve_device():
    """Pick CUDA when CTranslate2 can actually see a GPU."""

    if WHISPER_DEVICE:
        return WHISPER_DEVICE

    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"

    except Exception:
        pass

    return "cpu"


def preload_cuda_libraries():
    """
    Make cuBLAS 12 / cuDNN 9 from the nvidia-* pip wheels loadable.

    CTranslate2 opens them by soname, but pip puts them inside site-packages
    where the dynamic loader does not look. Loading them once with
    RTLD_GLOBAL lets the later lookups succeed without LD_LIBRARY_PATH.
    """

    try:
        import nvidia

    except ImportError:
        return

    libraries = []

    for base in nvidia.__path__:
        for pattern in ("cublas/lib/libcublas*.so.*", "cudnn/lib/libcudnn*.so.*"):
            libraries.extend(sorted(Path(base).glob(pattern)))

    # cuDNN sub-libraries depend on each other; retry until nothing changes
    while libraries:

        failed = []

        for library in libraries:
            try:
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)

            except OSError:
                failed.append(library)

        if len(failed) == len(libraries):
            break

        libraries = failed


def unload_ollama_models():
    """
    Free the GPU before Whisper runs.

    A previous run leaves the translator resident for LLM_KEEP_ALIVE, and
    Whisper large does not fit next to it on an 11 GB card.
    """

    try:
        client = ollama.Client()
        running = client.ps().models

    except Exception:
        return

    for loaded in running:

        try:
            client.generate(model=loaded.model, prompt="", keep_alive=0)
            print(f"Unloaded {loaded.model} from Ollama to free the GPU.")

        except Exception as error:
            print(f"   could not unload {loaded.model}: {error}")


def transcribe_video(video_path):
    """Run Whisper and return raw segments."""

    device = resolve_device()

    if device == "cuda":
        preload_cuda_libraries()
        unload_ollama_models()

    from faster_whisper import WhisperModel

    print()
    print("=" * 60)
    print(f"WHISPER: {WHISPER_MODEL} on {device} ({WHISPER_COMPUTE_TYPE})")
    print("=" * 60)

    model = WhisperModel(
        WHISPER_MODEL,
        device=device,
        compute_type=WHISPER_COMPUTE_TYPE,
    )

    print(f"\nTranscribing:\n{video_path}\n")

    results, info = model.transcribe(
        str(video_path),

        language=SOURCE_LANGUAGE,

        initial_prompt=WHISPER_INITIAL_PROMPT,

        # Do not feed hallucinated text back into the next window
        condition_on_previous_text=False,

        # Temperature fallback ladder. This is what lets Whisper escape a
        # repetition loop; a fixed temperature=0 keeps it stuck in one.
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],

        # Needed for hallucination_silence_threshold
        word_timestamps=True,

        # Skip stretches of silence the model tried to fill with invented text
        hallucination_silence_threshold=2.0,

        vad_filter=WHISPER_VAD_FILTER,

        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
    )

    # faster-whisper decodes lazily, so this loop is where the work happens
    segments = []

    for segment in results:

        segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "no_speech_prob": segment.no_speech_prob,
        })

        print(
            f"\r   {format_timestamp(segment.end)} / "
            f"{format_timestamp(info.duration)}",
            end="",
            flush=True,
        )

    print(f"\n\nWhisper produced {len(segments)} raw segments.")

    # Release the GPU memory before Ollama loads the translator, otherwise
    # Ollama has to spill layers to the CPU and translation slows down a lot
    del model, results
    gc.collect()

    return segments


def whisper_segments_to_entries(segments):
    """Whisper segments -> subtitle entries (uncleaned)."""

    entries = []

    for segment in segments:

        entries.append({
            "index": len(entries) + 1,
            "start": segment.get("start", 0.0),
            "end": segment.get("end", 0.0),
            "text": (segment.get("text") or "").strip(),
            "no_speech_prob": segment.get("no_speech_prob", 0.0),
        })

    return entries


# ============================================================
# TRANSLATION PROMPT
# ============================================================

def build_system_prompt():
    """
    Positive, task-shaped instructions.

    The previous version stacked twenty "NEVER ..." rules, which reads to a
    model as a description of the very output it should avoid. Structure is
    enforced by the JSON schema instead, so the prompt only has to describe
    good translation.
    """

    return f"""You are a professional subtitle translator working from \
{SOURCE_LANGUAGE_NAME} into {TARGET_LANGUAGE_NAME}.

About this video: {CONTENT_HINT}

You receive numbered subtitle lines and return one translation per line,
reusing the same id.

How to translate:
- Translate every line separately. Never merge, split, reorder or skip lines.
- {SOURCE_LANGUAGE_NAME} usually omits the subject and the pronouns. Use the
  surrounding lines to work out who is speaking, who they mean, and what is
  being referred to, then write {TARGET_LANGUAGE_NAME} that reads naturally
  in sequence with them.
- Write how people actually speak: contractions, fragments, interjections.
  A stilted literal gloss is a failed translation.
- Keep the speaker's register. Casual stays casual, crude stays crude,
  explicit stays explicit. Never censor, soften or summarise.
- Keep each line short enough to read on screen, roughly the length of the
  original.
- If a line is only a sound or filler with no meaning, give the closest
  natural {TARGET_LANGUAGE_NAME} equivalent.
- Write {TARGET_LANGUAGE_NAME} only: no {SOURCE_LANGUAGE_NAME} characters,
  no romaji, no notes, no alternatives, no explanations."""


TRANSLATION_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "translation": {"type": "string"},
                },
                "required": ["id", "translation"],
            },
        },
    },
    "required": ["lines"],
}


REFUSAL_PATTERN = re.compile(
    r"(?i)\b("
    r"i\s+(can'?t|cannot|won'?t|am\s+not\s+able)"
    r"|i'?m\s+(sorry|unable)"
    r"|as\s+an\s+ai"
    r"|i\s+do\s+not\s+feel\s+comfortable"
    r"|violates?\s+(my|the)\s+(guidelines|policy|policies)"
    r"|against\s+my\s+guidelines"
    r")\b"
)


def sanitize_translation(text):
    """Strip the artifacts a model still emits inside a JSON string field."""

    if not text:
        return ""

    text = normalize_text(text)

    # Reasoning blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"(?is)<think>.*$", "", text)
    text = re.sub(r"(?i)</?think>", "", text)

    # Labels
    text = re.sub(
        r"(?im)^\s*("
        r"translation|translated\s+text|english(\s+translation)?"
        r"|answer|response|output|subtitle|line\s*\d*"
        r")\s*[:\-]\s*",
        "",
        text,
    )

    # Invented SRT scaffolding
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    text = re.sub(
        r"\d{1,2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{3}",
        "",
        text,
    )

    text = text.replace("```", "")
    text = re.sub(r"(?m)^\s*>\s*", "", text)

    # Echoed source-language instructions
    text = strip_source_script(text)

    # Collapse to one logical line; wrapping happens at write time
    text = " ".join(line.strip() for line in text.splitlines() if line.strip())
    text = re.sub(r"\s{2,}", " ", text).strip()

    # Quotes wrapping the whole answer
    pairs = (('"', '"'), ("'", "'"), ("“", "”"), ("「", "」"))

    changed = True

    while changed and len(text) >= 2:

        changed = False

        for opening, closing in pairs:

            if text.startswith(opening) and text.endswith(closing):
                text = text[1:-1].strip()
                changed = True

    return text.strip()


def validate_translation(text):
    """Return usable subtitle text, or "" when the model output is unusable."""

    text = sanitize_translation(text)

    if not text:
        return ""

    # Untranslated or unsalvageable source-language output
    if has_source_script(text):
        return ""

    # Punctuation-only answers ("," was a real failure mode)
    if not re.search(r"[A-Za-z0-9]", text):
        return ""

    if len(text) > MAX_SUBTITLE_CHARS:
        return ""

    if REFUSAL_PATTERN.search(text):
        return ""

    return text


# ============================================================
# TRANSLATOR
# ============================================================

class Translator:
    """Batch translator with rolling context over an Ollama model."""

    def __init__(self, model=LLM_MODEL, client=None):

        self.model = model
        self.client = client or ollama.Client()
        self.system_prompt = build_system_prompt()

        # Not every model accepts think=False; detected on first use
        self.think_supported = True

        # [(source, translation), ...]
        self.context = []

    # --------------------------------------------------------

    def remember(self, source, translation):
        """Add a finished line to the context window."""

        self.context.append((source, translation))

        if len(self.context) > CONTEXT_LINES:
            self.context = self.context[-CONTEXT_LINES:]

    # --------------------------------------------------------

    def _build_messages(self, chunk):
        """chunk: list of (label, source_text)."""

        sections = []

        if self.context:

            previous = "\n".join(
                f"{source}  ->  {translation}"
                for source, translation in self.context
            )

            sections.append(
                "Earlier lines, already translated. Context only, do not "
                f"translate these again:\n{previous}"
            )

        numbered = "\n".join(
            f"{label}. {source}"
            for label, source in chunk
        )

        labels = ", ".join(str(label) for label, _ in chunk)

        sections.append(
            f"Translate these {len(chunk)} subtitle lines. "
            f"Return exactly {len(chunk)} objects, with ids {labels}:\n"
            f"{numbered}"
        )

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "\n\n".join(sections)},
        ]

    # --------------------------------------------------------

    def _chat(self, messages, temperature, num_predict):
        """One Ollama call with schema-constrained output."""

        kwargs = {
            "model": self.model,
            "messages": messages,
            "format": TRANSLATION_SCHEMA,
            "keep_alive": LLM_KEEP_ALIVE,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "repeat_penalty": 1.05,
                "num_ctx": LLM_NUM_CTX,
                "num_predict": num_predict,
            },
        }

        if self.think_supported:
            kwargs["think"] = False

        try:
            return self.client.chat(**kwargs)

        except Exception as error:

            # Models without a thinking mode reject the parameter outright
            if self.think_supported and "think" in str(error).lower():

                print("   note: model does not support think=False, dropping it.")

                self.think_supported = False
                kwargs.pop("think", None)

                return self.client.chat(**kwargs)

            raise

    # --------------------------------------------------------

    @staticmethod
    def _extract_content(response):
        """Pull the message content out of whichever response shape we get."""

        message = getattr(response, "message", None)

        if message is not None:
            return getattr(message, "content", "") or ""

        return (response or {}).get("message", {}).get("content", "") or ""

    # --------------------------------------------------------

    def _request(self, chunk, temperature):
        """
        Translate one chunk.

        Returns {label: translation} containing only the lines that came
        back valid.
        """

        messages = self._build_messages(chunk)

        num_predict = 120 * len(chunk) + 200

        try:
            response = self._chat(messages, temperature, num_predict)

        except Exception as error:
            print(f"   Ollama error: {error}")
            return {}

        content = self._extract_content(response)

        if not content.strip():
            return {}

        try:
            payload = json.loads(content)

        except json.JSONDecodeError:

            # The schema makes this unlikely, but a truncated response
            # can still land here.
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)

            if not match:
                print("   Model returned unparsable output.")
                return {}

            try:
                payload = json.loads(match.group(0))

            except json.JSONDecodeError:
                print("   Model returned unparsable output.")
                return {}

        lines = payload.get("lines")

        if not isinstance(lines, list):
            return {}

        expected = {label for label, _ in chunk}
        results = {}

        for item in lines:

            if not isinstance(item, dict):
                continue

            try:
                label = int(item.get("id"))

            except (TypeError, ValueError):
                continue

            if label not in expected:
                continue

            translation = validate_translation(item.get("translation", ""))

            if translation:
                results[label] = translation

        return results

    # --------------------------------------------------------

    def translate_batch(self, sources):
        """
        Translate a list of source lines.

        Always returns one translation per input line, in order. A line the
        model never handled falls back to its source text, so timings and
        numbering stay aligned with the video.
        """

        results = {}
        pending = list(range(len(sources)))

        for attempt in range(1, MAX_TRANSLATION_RETRIES + 1):

            if not pending:
                break

            temperature = TEMPERATURE_LADDER[
                min(attempt, len(TEMPERATURE_LADDER)) - 1
            ]

            if attempt > 1:
                print(
                    f"   Retry {attempt}/{MAX_TRANSLATION_RETRIES} "
                    f"for {len(pending)} line(s) at temperature {temperature}."
                )

            # Labels are 1-based positions within the batch
            chunk = [(position + 1, sources[position]) for position in pending]

            # From the second retry on, ask line by line: a single line is
            # much harder for the model to drift on.
            if attempt >= 3 and len(chunk) > 1:

                for label, source in chunk:

                    single = self._request([(label, source)], temperature)

                    if label in single:
                        results[label - 1] = single[label]

            else:

                for label, translation in self._request(chunk, temperature).items():
                    results[label - 1] = translation

            pending = [
                position
                for position in range(len(sources))
                if position not in results
            ]

        translations = []

        for position, source in enumerate(sources):

            translation = results.get(position)

            if not translation:
                print(f"   Falling back to source text for line {position + 1}.")
                translation = source

            translations.append(translation)

        return translations


# ============================================================
# TRANSLATE A WHOLE SRT
# ============================================================

def load_resume_state(output_path, entries):
    """
    Work out how many entries are already translated.

    Every input entry now produces exactly one output entry, so a valid
    partial file has contiguous indices 1..N. Anything else is from an older
    run and gets rewritten from scratch.
    """

    if FORCE_TRANSLATE or not output_path.exists():
        return 0, []

    try:
        existing = parse_srt(output_path)

    except Exception:
        return 0, []

    if not existing:
        return 0, []

    if len(existing) > len(entries):

        print(
            "\nExisting translation has more entries than the source "
            "transcript. Starting over."
        )

        return 0, []

    # Timings must line up with the current transcript, otherwise the old
    # file belongs to a different (e.g. uncleaned) transcript.
    for position, entry in enumerate(existing):

        if abs(entry["start"] - entries[position]["start"]) > 0.5:

            print(
                "\nExisting translation does not match the current "
                "transcript. Starting over."
            )

            return 0, []

    done = len(existing)

    context = [
        (entries[position]["text"], existing[position]["text"].replace("\n", " "))
        for position in range(max(0, done - CONTEXT_LINES), done)
    ]

    print(f"\nResuming: {done}/{len(entries)} lines already translated.")

    return done, context


def translate_srt(entries, output_path, model=LLM_MODEL):
    """Translate every entry and write the target SRT."""

    done, context = load_resume_state(output_path, entries)

    if done >= len(entries) and entries:
        print("\nTranslation already complete.")
        return

    if FORCE_TRANSLATE and output_path.exists():
        print("\nFORCE_TRANSLATE is on: deleting the existing translation.")
        output_path.unlink()

    print()
    print("=" * 60)
    print(f"OLLAMA: {model}")
    print("=" * 60)

    translator = Translator(model=model)
    translator.context = context

    mode = "a" if done else "w"

    with open(output_path, mode, encoding="utf-8") as handle:

        position = done

        while position < len(entries):

            batch = entries[position:position + BATCH_SIZE]
            sources = [entry["text"] for entry in batch]

            print()
            print("=" * 60)
            print(
                f"[{position + 1}-{position + len(batch)}"
                f"/{len(entries)}]"
            )

            translations = translator.translate_batch(sources)

            for offset, (entry, translation) in enumerate(zip(batch, translations)):

                print(f"  JP: {entry['text']}")
                print(f"  EN: {translation}")

                write_srt_entry(
                    handle,
                    position + offset + 1,
                    entry["start"],
                    entry["end"],
                    wrap_subtitle(translation),
                )

                translator.remember(entry["text"], translation)

            position += len(batch)

    print()
    print("=" * 60)
    print("TRANSLATION FINISHED")
    print("=" * 60)
    print(f"\nSaved to:\n{output_path}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def generate_subtitles(
    video_path,
    model=LLM_MODEL,
    force_whisper=None,
    force_translate=None,
):

    global FORCE_WHISPER, FORCE_TRANSLATE

    if force_whisper is not None:
        FORCE_WHISPER = force_whisper

    if force_translate is not None:
        FORCE_TRANSLATE = force_translate

    video_path = Path(video_path).expanduser().resolve()

    if not video_path.exists():
        print(f"Video not found:\n{video_path}")
        return False

    source_srt = video_path.with_suffix(".srt")

    target_srt = video_path.with_name(
        f"{video_path.stem}.{TARGET_LANGUAGE_NAME[:2].lower()}.srt"
    )

    print()
    print("=" * 60)
    print("SUBTITLE TRANSLATOR")
    print("=" * 60)
    print(f"\nVideo:\n{video_path}")
    print(f"\n{SOURCE_LANGUAGE_NAME} SRT:\n{source_srt}")
    print(f"\n{TARGET_LANGUAGE_NAME} SRT:\n{target_srt}")

    # ------------------------------------------------------------
    # Transcript: reuse an existing one, or run Whisper
    # ------------------------------------------------------------

    if source_srt.exists() and not FORCE_WHISPER:

        print(f"\nUsing the existing {SOURCE_LANGUAGE_NAME} SRT.")

        entries = clean_entries(parse_srt(source_srt))

    else:

        print("\nNo transcript yet, running Whisper.")

        segments = transcribe_video(video_path)

        entries = clean_entries(whisper_segments_to_entries(segments))

        if entries:
            write_srt(source_srt, entries)
            print(f"\n{SOURCE_LANGUAGE_NAME} SRT saved to:\n{source_srt}")

    if not entries:
        print("\nNo usable subtitles found.")
        return False

    print(f"\nUsable subtitles: {len(entries)}")

    # ------------------------------------------------------------
    # Translate
    # ------------------------------------------------------------

    translate_srt(entries, target_srt, model=model)

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\n{TARGET_LANGUAGE_NAME} subtitles:\n{target_srt}")

    return True


# ============================================================
# CLI
# ============================================================

def main(argv=None):

    global WHISPER_MODEL, BATCH_SIZE, CONTENT_HINT, DROP_VOCALIZATIONS

    parser = argparse.ArgumentParser(
        description="Transcribe a video with Whisper and translate it with Ollama."
    )

    parser.add_argument(
        "video",
        nargs="?",
        default=VIDEO_FILE,
        help="path to the video file",
    )

    parser.add_argument(
        "--model",
        default=LLM_MODEL,
        help=f"Ollama model to translate with (default: {LLM_MODEL})",
    )

    parser.add_argument(
        "--whisper-model",
        default=WHISPER_MODEL,
        help=f"Whisper model size (default: {WHISPER_MODEL})",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"subtitles per translation request (default: {BATCH_SIZE})",
    )

    parser.add_argument(
        "--context-hint",
        default=CONTENT_HINT,
        help="one sentence describing the video, given to the translator",
    )

    parser.add_argument(
        "--keep-sounds",
        action="store_true",
        help="keep non-verbal lines (moans, gasps) instead of dropping them",
    )

    parser.add_argument(
        "--force-whisper",
        action="store_true",
        help="re-transcribe even if a source SRT exists",
    )

    parser.add_argument(
        "--force-translate",
        action="store_true",
        help="re-translate from scratch instead of resuming",
    )

    args = parser.parse_args(argv)

    if not args.video:
        parser.error("no video given: pass a path, or set VIDEO_FILE in the script")

    WHISPER_MODEL = args.whisper_model
    BATCH_SIZE = max(1, args.batch_size)
    CONTENT_HINT = args.context_hint
    DROP_VOCALIZATIONS = not args.keep_sounds

    ok = generate_subtitles(
        args.video,
        model=args.model,
        force_whisper=args.force_whisper,
        force_translate=args.force_translate,
    )

    return 0 if ok else 1


if __name__ == "__main__":
    #video file path
    video_file = "/home/gato/Videos/HHGsdDsh_720p.mp4"
    generate_subtitles(video_file)