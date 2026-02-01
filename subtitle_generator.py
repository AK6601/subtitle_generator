import whisper
import os
import ollama
import datetime
import re

def format_timestamp(seconds):
    """Converts seconds to SRT timestamp format"""
    x = datetime.timedelta(seconds=float(seconds))
    total_seconds = int(x.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    milliseconds = int(x.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def clean_llm_output(text):
    """Cleans the LLM output (removes 'Translation:', brackets, etc.)"""
    text = re.sub(r"[\(\[].*?[\)\]]", "", text)
    text = re.sub(r"(?i)^translation[:\-\s]*", "", text)
    text = re.sub(r"(?i)^translated text[:\-\s]*", "", text)
    text = text.strip().strip('"').strip("'")
    return text

def generate_subtitles(video_path, whisper_model="large", llm_model="dolphin-llama3"): # 
    """
    whisper: whisper_model="medium" also works 
    change llm_model to desired model from ollama, i choose dolphin because its realtivly uncensored 
    """
    
    if not os.path.exists(video_path):
        print(f"Error: File '{video_path}' not found.")
        return

    base_name = os.path.splitext(video_path)[0]
    srt_output_file = f"{base_name}.srt"

    print(f"\n1. 👂 Listening with Whisper ({whisper_model})...")
    model = whisper.load_model(whisper_model, device="cpu")
    
    audio_result = model.transcribe(
        video_path, 
        language="ja", # change to language of video
        condition_on_previous_text=False, 
        fp16=False,
        word_timestamps=True 
    )
    segments = audio_result["segments"]
    
    print(f"--- Extracted {len(segments)} segments. Filtering & translating... ---")

    with open(srt_output_file, "w", encoding="utf-8") as srt_file:
        # change as you see fit (prompt for LLM)
        system_instruction = (
            "You are a subtitle generator. "
            "Translate the Japanese text to English immediately. "
            "Do not start with 'Translation:'. "
            "Do not add notes. "
            "Output ONLY the English text."
        )

        subtitle_index = 1

        for segment in segments:
            start_time = format_timestamp(segment['start'])
            end_time = format_timestamp(segment['end'])
            japanese_text = segment['text'].strip()

            # --- FILTER 1: Skip Empty/Tiny Segments ---
            if not japanese_text or len(japanese_text) < 2:
                continue
            
            # Send to Ollama
            response = ollama.chat(model=llm_model, messages=[
                {'role': 'system', 'content': system_instruction},
                {'role': 'user', 'content': japanese_text},
            ])

            clean_english = clean_llm_output(response['message']['content'])

            if not clean_english:
                continue
            
            # Write SRT block
            srt_block = f"{subtitle_index}\n{start_time} --> {end_time}\n{clean_english}\n\n"
            srt_file.write(srt_block)
            srt_file.flush()
            
            print(f"[{subtitle_index}] {start_time}: {japanese_text} -> {clean_english}")
            subtitle_index += 1

    print(f"\n=== SUCCESS ===\nSaved subtitles to: {srt_output_file}")

if __name__ == "__main__":
    #video file path
    video_file = "/path/to/video"
    generate_subtitles(video_file)
