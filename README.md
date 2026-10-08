# AI-Powered Subtitle Generator & Translator

A local Python utility that automates the creation of translated subtitles (`.srt`) for video files. This tool combines **Whisper** (via [faster-whisper](https://github.com/SYSTRAN/faster-whisper)) for speech-to-text and **Ollama** for LLM-driven translation.

## 🚀 Features

* **Local:** Everything runs on your machine — no API keys or cloud costs.
* **Context-aware translation:** Subtitles are translated in batches, with the previously translated lines supplied as context. Japanese drops subjects and pronouns constantly, so line-by-line translation produces gibberish; the model needs the surrounding dialogue.
* **Schema-constrained output:** Ollama is called with a JSON schema, so the model physically cannot emit reasoning, labels, timestamps or source-language text into the SRT.
* **Transcript cleanup:** Whisper repetition loops are merged, non-verbal sounds are dropped, and subtitle durations are capped.
* **No dropped lines:** Every input subtitle produces exactly one output subtitle. If translation fails after all retries the source text is kept, so timings and numbering never drift out of sync with the video.
* **Resumable:** Interrupt at any time and re-run; the translation continues where it stopped.

## 🛠️ Prerequisites

1. **Python 3.8+**
2. **NVIDIA driver** with CUDA 12 support (optional, but Whisper on CPU is very slow). The CUDA libraries themselves come from pip.
3. **Ollama:** [Download here](https://ollama.com), then pull a model:
   ```bash
   ollama pull richardyoung/qwen3-14b-abliterated
   ```

## 📦 Installation

Use a dedicated virtual environment: the CUDA 12 libraries faster-whisper needs would overwrite the CUDA 13 ones a recent PyTorch installs into the same `site-packages/nvidia` folders.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run the script with `.venv/bin/python`, or select `.venv` as the interpreter in your IDE. The script loads the pip-installed CUDA libraries itself, no `LD_LIBRARY_PATH` needed.

## 🖥️ Usage

```bash
python subtitle_generator.py /path/to/video.mp4
```

Useful flags:

```bash
# describe the video — the single most effective quality knob
python subtitle_generator.py video.mp4 --context-hint "Two friends arguing in a kitchen."

# try a different translator
python subtitle_generator.py video.mp4 --model hermes3:8b

# re-transcribe / re-translate from scratch
python subtitle_generator.py video.mp4 --force-whisper --force-translate

# keep moans, gasps and other non-verbal lines
python subtitle_generator.py video.mp4 --keep-sounds
```

Output lands next to the source video: `video.srt` (transcript) and `video.en.srt` (translation).

## ⚙️ Configuration

Command-line flags cover the common cases; everything else lives in the `CONFIGURATION` block at the top of the script.

| Setting | Description | Default |
| :--- | :--- | :--- |
| `WHISPER_MODEL` | Whisper size (`tiny`…`large`). | `large` |
| `WHISPER_DEVICE` | `None` auto-detects CUDA, falls back to CPU. | `None` |
| `WHISPER_COMPUTE_TYPE` | `float32` (full quality), `int8_float32` (faster), `float16` (Volta+ only). | `float32` |
| `WHISPER_VAD_FILTER` | Skip silence before transcribing. Faster, fewer hallucinations, different line splits. | `False` |
| `LLM_MODEL` | Ollama model used for translation. | `richardyoung/qwen3-14b-abliterated` |
| `SOURCE_LANGUAGE` | Spoken language of the video (ISO code). | `ja` |
| `CONTENT_HINT` | One sentence about the video, given to the translator. | — |
| `BATCH_SIZE` | Subtitles per translation request. | `8` |
| `CONTEXT_LINES` | Already-translated lines shown as context. | `4` |
| `MAX_TRANSLATION_RETRIES` | Batch retries before falling back to line-by-line. | `3` |
| `DROP_VOCALIZATIONS` | Drop lines that are only sounds. | `True` |
| `COLLAPSE_REPEATS` | Merge consecutive identical segments. | `True` |
| `MAX_SUBTITLE_DURATION` | Seconds a subtitle may stay on screen. | `8.0` |

## 📄 How it Works

1. **Transcription:** Whisper transcribes the audio into timestamped segments. It runs through faster-whisper (CTranslate2) on the GPU; any model still loaded in Ollama is unloaded first, and Whisper is released afterwards, so the translator gets the whole card. A temperature fallback ladder and `hallucination_silence_threshold` are enabled — these are what let Whisper escape the repetition loops that otherwise fill silent stretches with the same line dozens of times.

2. **Cleanup:** `clean_entries` merges adjacent duplicates, drops non-verbal sounds and watermark text, caps subtitle duration and renumbers. This runs on an existing transcript too, so re-running cleans up an older bad `.srt`.

3. **Translation:** Subtitles are sent to Ollama in batches, numbered, together with the last few translated lines as context. The response is constrained to a JSON schema (`{"lines": [{"id", "translation"}]}`), which grammar-constrains decoding and removes any need to regex prose out of the answer.

4. **Validation:** Each translation is checked for source-language leakage (echoed instructions are salvaged where possible), refusals, punctuation-only answers and excessive length. Failures retry at a higher temperature, then line-by-line, then fall back to the source text.

5. **Assembly:** Lines are wrapped to two screen-width lines and appended to the SRT as they are produced, so progress survives an interrupt.

## 🩺 Troubleshooting

**Translations are literal or nonsensical.** Check the source `.srt` first — if the Japanese itself is wrong, no translator can fix it. Re-run with `--force-whisper`, and use a larger Whisper model.

**Lines fall back to the source text.** The model refused, or kept answering in the source language. Try an uncensored model (`--model`), or lower `--batch-size`.

**Whisper is very slow.** It is running on CPU — the header line says `on cpu`. Check that you run the script from `.venv` and that `nvidia-smi` works. Without a GPU, `--whisper-model medium` is a good compromise.

**`TypeError: open() got an unexpected keyword argument 'metadata_errors'`.** PyAV is too new for faster-whisper: `pip install "av<16"`.
