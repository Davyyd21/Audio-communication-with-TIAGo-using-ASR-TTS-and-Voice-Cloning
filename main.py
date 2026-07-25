import argparse
from pathlib import Path

from tiago_assistant.asr import ASR
from tiago_assistant.context_selector import ContextSelector
from tiago_assistant.conversation import Conversation
from tiago_assistant.dialog import Dialog
from tiago_assistant.prompt_builder import PromptBuilder
from tiago_assistant.recorder import AudioRecorder
from tiago_assistant.retriever import Retriever, SearchResult


def parse_arguments() -> argparse.Namespace:
    """
    Citește opțiunile primite din terminal.

    Exemplu:
        python main.py --model base --language ro --duration 7
    """

    parser = argparse.ArgumentParser(
        description=(
            "Record microphone audio, transcribe it with Whisper, "
            "retrieve laboratory information and generate a "
            "Gemini response."
        )
    )

    parser.add_argument(
        "--model",
        default="base",
        help="Whisper model name. Default: base.",
    )

    parser.add_argument(
        "--language",
        default="ro",
        help="Audio and response language. Default: ro.",
    )

    parser.add_argument(
        "--knowledge",
        default="knowledge",
        help=(
            "Path to the knowledge directory. "
            "Default: knowledge."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help=(
            "Maximum number of primary retrieved fragments. "
            "Default: 3."
        ),
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=7.0,
        help=(
            "Duration of each microphone recording in seconds. "
            "Default: 7."
        ),
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help=(
            "Microphone sample rate. "
            "Default: 16000 Hz."
        ),
    )

    parser.add_argument(
        "--output",
        default="samples/input/live_question.wav",
        help=(
            "Path used for the current microphone recording. "
            "The file is overwritten after every question."
        ),
    )

    return parser.parse_args()


def print_search_results(
    results: list[SearchResult],
) -> None:
    """
    Afișează fragmentele selectate de Retriever.
    """

    if not results:
        print("\nNo relevant fragments found.")
        return

    print("\nRetrieved fragments:")

    for result in results:
        print(
            f"- {result.source} "
            f"[fragment {result.position + 1}] "
            f"(score: {result.score:.2f})"
        )


def retrieve_context(
    question: str,
    active_laboratory: str | None,
    retriever: Retriever,
    top_k: int = 3,
) -> list[SearchResult]:
    """
    Selectează fragmentele care trebuie trimise către Gemini.

    Strategia:

    1. Pentru o prezentare generală sunt luate toate
       fragmentele laboratorului activ.

    2. Pentru o întrebare specifică sunt căutate fragmentele
       cele mai relevante din laboratorul activ.

    3. Dacă nu există rezultate, sunt folosite toate
       fragmentele laboratorului activ.

    4. Dacă nu există laborator activ, căutarea se face
       în toate fișierele.
    """

    if active_laboratory is not None:
        if retriever.is_general_presentation_question(
            question
        ):
            return retriever.get_laboratory_chunks(
                laboratory_name=active_laboratory,
                max_chunks=10,
            )

        laboratory_results = retriever.search(
            question=question,
            laboratory_name=active_laboratory,
            top_k=top_k,
            include_neighbors=True,
        )

        if laboratory_results:
            return laboratory_results

        laboratory_fallback = (
            retriever.get_laboratory_chunks(
                laboratory_name=active_laboratory,
                max_chunks=10,
            )
        )

        if laboratory_fallback:
            return laboratory_fallback

    return retriever.search(
        question=question,
        laboratory_name=None,
        top_k=top_k,
        include_neighbors=True,
    )


def process_transcription(
    transcription: str,
    language: str,
    top_k: int,
    context_selector: ContextSelector,
    conversation: Conversation,
    retriever: Retriever,
    prompt_builder: PromptBuilder,
    dialog: Dialog,
) -> None:
    """
    Procesează o întrebare deja transcrisă.

    Flux:
        text
        -> detectare laborator
        -> Retriever
        -> PromptBuilder
        -> Gemini
        -> salvare în Conversation
    """

    cleaned_transcription = transcription.strip()

    if not cleaned_transcription:
        print(
            "\nWhisper did not detect any usable speech."
        )
        return

    print("\nTranscription:")
    print(cleaned_transcription)

    detected_laboratory, _ = (
        context_selector.get_context(
            cleaned_transcription
        )
    )

    if detected_laboratory is not None:
        conversation.set_active_laboratory(
            detected_laboratory
        )

    active_laboratory = (
        conversation.get_active_laboratory()
    )

    print("\nActive laboratory:")

    if active_laboratory is None:
        print("None")
    else:
        print(active_laboratory)

    results = retrieve_context(
        question=cleaned_transcription,
        active_laboratory=active_laboratory,
        retriever=retriever,
        top_k=top_k,
    )

    print_search_results(results)

    relevant_context = (
        retriever.format_results(results)
    )

    prompt = prompt_builder.build(
        question=cleaned_transcription,
        laboratory_name=active_laboratory,
        laboratory_context=relevant_context,
        conversation_history=(
            conversation.get_messages()
        ),
        language=language,
    )

    print("\nSending prompt to Gemini...")

    try:
        answer = dialog.generate_response(
            prompt
        )

    except RuntimeError as error:
        print("\nGemini request failed:")
        print(error)
        return

    if not answer or not answer.strip():
        print(
            "\nGemini did not return a usable answer."
        )
        return

    cleaned_answer = answer.strip()

    conversation.add_user_message(
        cleaned_transcription
    )

    conversation.add_assistant_message(
        cleaned_answer
    )

    print("\nTIAGo response:")
    print(cleaned_answer)


def main() -> None:
    """
    Rulează conversația audio continuă.

    La fiecare iterație:

        utilizatorul apasă Enter
            ↓
        microfonul înregistrează
            ↓
        Whisper transcrie
            ↓
        Retriever caută informații
            ↓
        Gemini răspunde
            ↓
        conversația este păstrată
            ↓
        programul așteaptă următoarea întrebare
    """

    args = parse_arguments()

    if args.duration <= 0:
        raise ValueError(
            "Recording duration must be greater than zero."
        )

    if args.top_k <= 0:
        raise ValueError(
            "Top-k must be greater than zero."
        )

    print("Initializing TIAGo assistant...")

    recorder = AudioRecorder(
        sample_rate=args.sample_rate,
        channels=1,
    )

    asr = ASR(
        model_name=args.model
    )

    context_selector = ContextSelector(
        args.knowledge
    )

    conversation = Conversation(
        max_messages=10
    )

    retriever = Retriever(
        knowledge_directory=args.knowledge,
        minimum_score=0.18,
    )

    prompt_builder = PromptBuilder()
    dialog = Dialog()

    recording_path = Path(
        args.output
    )

    print("\nTIAGo assistant started.")
    print(
        "Press Enter to record a question."
    )
    print(
        "Commands: exit, reset, laboratory\n"
    )

    while True:
        try:
            command = input(
                "Press Enter to record: "
            ).strip().lower()

        except (KeyboardInterrupt, EOFError):
            print("\nConversation ended.")
            break

        if command in {
            "exit",
            "quit",
            "stop",
            "iesire",
            "ieșire",
        }:
            print("Conversation ended.")
            break

        if command in {
            "reset",
            "restart",
        }:
            conversation.clear()

            print(
                "\nConversation and active laboratory "
                "were reset.\n"
            )

            continue

        if command in {
            "laboratory",
            "laborator",
        }:
            active_laboratory = (
                conversation.get_active_laboratory()
            )

            if active_laboratory is None:
                print(
                    "\nNo laboratory is currently active.\n"
                )
            else:
                print(
                    f"\nActive laboratory: "
                    f"{active_laboratory}\n"
                )

            continue

        if command:
            print(
                "\nUnknown command. Press Enter without "
                "typing anything to record.\n"
            )
            continue

        try:
            audio_path = recorder.record(
                output_path=recording_path,
                duration=args.duration,
            )

        except RuntimeError as error:
            print("\nRecording failed:")
            print(error)
            print()
            continue

        print("\nTranscribing audio...")

        try:
            transcription = asr.transcribe(
                audio_path=str(audio_path),
                language=args.language,
            )

        except RuntimeError as error:
            print("\nWhisper transcription failed:")
            print(error)
            print()
            continue

        process_transcription(
            transcription=transcription,
            language=args.language,
            top_k=args.top_k,
            context_selector=context_selector,
            conversation=conversation,
            retriever=retriever,
            prompt_builder=prompt_builder,
            dialog=dialog,
        )

        print()


if __name__ == "__main__":
    main()