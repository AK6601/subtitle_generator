# AI-Powered Subtitle Generator & Translator

A local Python utility that automates the creation of translated subtitles (`.srt`) for video files. This tool combines **OpenAI's Whisper** for robust speech-to-text and **Ollama** for nuanced, LLM-driven translation.

## 🚀 Features

* **Local Transcription:** Everything runs on your machine—no API keys or cloud costs required.
* **High Accuracy:** Uses Whisper's `large` model for precise transcription.
* **Contextual Translation:** Leverages Ollama to handle slang and context better than traditional translators.
* **Automatic Cleaning:** Built-in regex filters to remove LLM conversational filler and formatting artifacts.
* **Standard Output:** Generates ready-to-use `.srt` files compatible with VLC and other players.

## 🛠️ Prerequisites

Before running the script, ensure you have the following installed:

1.  **Python 3.8+**
2.  **FFmpeg:** Required by Whisper for audio extraction.
3.  **Ollama:** [Download here](https://ollama.com).
    * Pull your preferred model: `ollama pull dolphin-llama3`

## 📦 Installation

1. Clone this repository or save the script
2. Install the required Python libraries via pip:

```bash
pip install openai-whisper ollama
```

## ⚙️ Configuration
You can customize the script by editing the generate_subtitles parameters or the variables within the script:

| Parameter | Description | Default |
| :--- | :--- | :--- |
| **video_path** | Path to your source video file. | Required |
| **whisper_model** | Whisper model size (`tiny`, `base`, `medium`, `large`). | `large` |
| **llm_model** | The model name as it appears in Ollama. | `dolphin-llama3` |
| **language** | The source language of the video (ISO code). | `ja` (Japanese) |

## 🖥️ Usage

1. **Set your file path:** Open the script and update the `video_file` variable at the bottom:

   ```python
   if __name__ == "__main__":
       video_file = "/path/to/your/video.mp4"
       generate_subtitles(video_file)
    ```

2. **Run the script:**
```bash
python subtitle_generator.py
```

3. **Result:** The script will output a ```.srt``` file in the same directory as your source video.

## 📄 How it Works

The script follows a linear pipeline to ensure high-quality subtitle blocks:

1. **Transcription:** **Whisper** processes the video audio, breaking it into segments with millisecond-precise timestamps.
   
2. **Translation:** Each text segment is passed to **Ollama**. A specific system prompt instructs the LLM to act as a subtitle generator, translating the text immediately without adding conversational notes.

3. **Refinement:** The `clean_llm_output` function uses regular expressions to strip out unwanted artifacts like "Translation:", "Translated text:", or text inside brackets.

4. **Assembly:** The script formats the cleaned translation and the Whisper timestamps into the **SubRip (SRT)** format and saves it to disk.
