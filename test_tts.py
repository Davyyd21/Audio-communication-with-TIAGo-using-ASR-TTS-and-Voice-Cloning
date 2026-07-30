from pathlib import Path

from tiago_assistant.tts import XTTS


def main() -> None:
    tts = XTTS()

    tts.synthesize(
        text=(
            "Hello! This is a test of the voice cloning system "
            "for the TIAGo robot."
        ),
        reference_audio=Path("samples/reference/david.wav"),
        output_audio=Path("samples/output/answer.wav"),
        language="en",
    )


if __name__ == "__main__":
    main()