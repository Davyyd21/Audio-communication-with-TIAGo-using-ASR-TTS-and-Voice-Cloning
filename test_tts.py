from pathlib import Path

from tiago_assistant.tts import RomanianTTS


def main() -> None:
    tts = RomanianTTS(
        model_path=Path(
            "models/piper/ro_RO-lili-high.onnx"
        ),
    )

    tts.synthesize(
        text=(
            "Bună ziua! Eu sunt robotul Tiago și vă pot ajuta să descoperiți "
            "laboratoarele facultății, proiectele dezvoltate de studenți și "
            "echipamentele folosite pentru cercetare. Pot asculta întrebările "
            "dumneavoastră, pot înțelege informațiile importante și pot răspunde "
            "clar, folosind limba română."
        ),
        output_audio=Path(
            "samples/output/answer.wav"
        ),
    )


if __name__ == "__main__":
    main()