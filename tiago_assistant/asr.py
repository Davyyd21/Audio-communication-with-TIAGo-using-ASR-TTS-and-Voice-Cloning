from pathlib import Path

import whisper


class ASR:
    def __init__(self, model_name: str = "base") -> None:
        print(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_path: str | Path, language: str = "ro") -> str:
        path = Path(audio_path)

        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        result = self.model.transcribe(
            str(path),
            language=language,
            task="transcribe", #whisper-ul trebuie setat ori sa faca transcribe ori sa faca translate
            fp16=False, #aparent fp16 este doar pentru rularea pe GPU
        )

        text = result["text"].strip() #elimina spatiile goale dintr-un sir de caractere asa " wow " -> "wow"

        if not text:
            raise ValueError("Whisper returned an empty transcription.")

        return text