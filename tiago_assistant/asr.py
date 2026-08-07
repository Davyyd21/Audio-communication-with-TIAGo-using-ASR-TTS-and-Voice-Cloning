from pathlib import Path
import re

import whisper


class ASR:
    """
    Încarcă modelul Whisper o singură dată și îl reutilizează
    pentru transcrierea fișierelor audio în limba română.
    """

    def __init__(
        self,
        model_name: str = "base",
    ) -> None:
        print(f"Loading Whisper model: {model_name}")

        self.model = whisper.load_model(
            model_name
        )

    def transcribe(
        self,
        audio_path: str | Path,
        language: str = "ro",
    ) -> str:
        """
        Transcrie un fișier audio în limba specificată.
        """

        path = Path(audio_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Audio file not found: {path}"
            )

        result = self.model.transcribe(
            str(path),

            # Forțează limba română.
            language=language,

            # Transcrie în aceeași limbă, nu traduce.
            task="transcribe",

            # Rulare pe CPU.
            fp16=False,

            # Reduce repetările și propagarea greșelilor
            # între ferestrele audio.
            condition_on_previous_text=False,

            # Oferă context despre vocabularul probabil.
            initial_prompt=(
                "Conversație în limba română despre "
                "laboratoare universitare, robotică, "
                "inteligență artificială, procesarea semnalelor, "
                "recunoaștere vocală, sisteme integrate, "
                "rețele inteligente, Tiago, lidar, FPGA "
                "și echipamente de laborator."
            ),

            # Folosește decodare deterministă.
            temperature=0.0,

            # Ignoră mai ușor porțiunile considerate fără vorbire.
            no_speech_threshold=0.6,

            # Controlează filtrarea segmentelor foarte improbabile.
            logprob_threshold=-1.0,

            # Filtrează segmentele cu repetiții anormale.
            compression_ratio_threshold=2.4,
        )

        text = str(
            result.get("text", "")
        ).strip()

        text = self._clean_romanian_text(
            text
        )

        if not text:
            raise ValueError(
                "Whisper returned an empty transcription."
            )

        return text

    @staticmethod
    def _clean_romanian_text(
        text: str,
    ) -> str:
        """
        Elimină simbolurile și caracterele care nu sunt utile
        pentru o transcriere în limba română.

        Nu corectează cuvintele transcrise greșit.
        """

        cleaned_text = re.sub(
            r"[^A-Za-zĂÂÎȘȚăâîșț0-9\s.,?!:;\-]",
            " ",
            text,
        )

        cleaned_text = re.sub(
            r"\s+",
            " ",
            cleaned_text,
        )

        return cleaned_text.strip()