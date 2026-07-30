from pathlib import Path

from TTS.api import TTS


class XTTS:
    """
    Transformă un text într-un fișier WAV folosind XTTS-v2
    și o înregistrare audio de referință pentru voice cloning.
    """

    MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

    def __init__(self) -> None:
        """
        Încarcă modelul XTTS-v2 pe CPU.
        """

        print("Loading XTTS-v2 model on CPU...")

        try:
            self.model = TTS(
                model_name=self.MODEL_NAME,
                progress_bar=True,
            ).to("cpu")

        except Exception as error:
            raise RuntimeError(
                "XTTS-v2 model could not be loaded."
            ) from error

    def synthesize(
        self,
        text: str,
        reference_audio: str | Path,
        output_audio: str | Path,
        language: str = "ro",
    ) -> Path:
        """
        Generează un fișier WAV folosind textul primit
        și vocea din fișierul audio de referință.
        """

        cleaned_text = text.strip()

        if not cleaned_text:
            raise ValueError(
                "The text sent to XTTS cannot be empty."
            )

        reference_path = Path(reference_audio)
        output_path = Path(output_audio)

        if not reference_path.exists():
            raise FileNotFoundError(
                f"Reference audio was not found: {reference_path}"
            )

        if not reference_path.is_file():
            raise ValueError(
                f"Reference audio path is not a file: {reference_path}"
            )

        if reference_path.suffix.lower() != ".wav":
            raise ValueError(
                "The reference audio must be a WAV file."
            )

        if output_path.suffix.lower() != ".wav":
            raise ValueError(
                "The output audio must have the .wav extension."
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print("\nGenerating speech with XTTS-v2...")
        print(f"Reference audio: {reference_path}")
        print(f"Output audio: {output_path}")

        try:
            self.model.tts_to_file(
                text=cleaned_text,
                speaker_wav=str(reference_path),
                language=language,
                file_path=str(output_path),
            )

        except Exception as error:
            raise RuntimeError(
                "XTTS-v2 speech generation failed."
            ) from error

        if not output_path.exists():
            raise RuntimeError(
                "XTTS finished without creating the output WAV file."
            )

        print(f"Generated audio saved to: {output_path}")

        return output_path