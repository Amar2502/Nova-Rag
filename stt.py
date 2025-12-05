from faster_whisper import WhisperModel
import tempfile

whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")

def transcribe_chunk(file_path):
    """Convert audio file to text using Whisper model."""

    segments, _ = whisper_model.transcribe(
        file_path,
        beam_size=5,
        language="en",
    )
    transcribed_text = " ".join([segment.text for segment in segments]).strip()

    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode="w", encoding="utf-8") as txt_file:
        txt_file.write(transcribed_text)
        tmp_text_path = txt_file.name
        
    return tmp_text_path