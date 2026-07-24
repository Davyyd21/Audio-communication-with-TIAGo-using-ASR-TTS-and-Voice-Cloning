import argparse
from pathlib import Path

from tiago_assistant.asr import ASR
from tiago_assistant.context_selector import ContextSelector

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TIAGo voice assistant laptop application."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input audio file.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
        help="Whisper model used for transcription.",
    )

    parser.add_argument(
        "--language",
        type=str,
        default="ro",
        help="Language code of the input audio, for example ro or en.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    asr = ASR(model_name=args.model)

    transcription = asr.transcribe(
        audio_path=args.input,
        language=args.language,
    )

    print("\nTranscription:")
    print(transcription)

    selector=ContextSelector("knowledge")
    laboratory_name, laboratory_context = selector.get_context(transcription)

    if laboratory_name is None:
        print("\nLaboratory detected:")
        print("None")

        print("\nKnowledge loaded:")
        print("No laboratory context was found.")
        return

    print("\nLaboratory detected:")
    print(laboratory_name)

    print("\nKnowledge loaded:")
    print(laboratory_context)

if __name__ == "__main__":
    main()