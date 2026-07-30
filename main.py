import argparse
from pathlib import Path

from tiago_assistant.asr import ASR
from tiago_assistant.context_selector import ContextSelector
from tiago_assistant.conversation import Conversation
from tiago_assistant.dialog import Dialog
from tiago_assistant.prompt_builder import PromptBuilder
from tiago_assistant.recorder import AudioRecorder
from tiago_assistant.retriever import Retriever, SearchResult
from tiago_assistant.tts import XTTS


def parse_arguments() -> argparse.Namespace:
    """
    citeste optiunile primite din terminal
    Ex:
        python main.py --model base --language ro --duration 7
    """

    parser = argparse.ArgumentParser(
        description=(
            "Record microphone audio, transcribe it with Whisper, "
            "retrieve laboratory information, generate a Gemini "
            "response and synthesize it with XTTS."
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
        help=(
            "Language spoken by the user and used by Whisper. "
            "Default: ro."
        ),
    )

    parser.add_argument(
        "--response-language",
        default="en",
        help=(
            "Language used by Gemini for the response. "
            "XTTS-v2 supports English. Default: en."
        ),
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

    parser.add_argument(
        "--reference-audio",
        default="samples/reference/david.wav",
        help=(
            "Reference WAV file used by XTTS for voice cloning. "
            "Default: samples/reference/david.wav."
        ),
    )

    parser.add_argument(
        "--tts-output",
        default="samples/output/answer.wav",
        help=(
            "Path used for the XTTS response. "
            "The file is overwritten after every answer."
        ),
    )

    return parser.parse_args()


def print_search_results(
    results: list[SearchResult],
) -> None:
    """
    afiseaza fragmentele selectate de Retriever
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
    selecteaza fragmentele care trebuie trimise catre Gemini.

    practic facem asa:

    1. Pentru o prezentare generala sunt luate toate
       fragmentele laboratorului activ.
    2. Pentru o intrebare specifica sunt cautate fragmentele
       cele mai relevante din laboratorul activ.
    3. Daca nu exista rezultate, sunt folosite toate
       fragmentele laboratorului activ.
    4. Daca nu exista laborator activ, cautarea se face
       in toate fisierele.
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
    response_language: str,
    top_k: int,
    context_selector: ContextSelector,
    conversation: Conversation,
    retriever: Retriever,
    prompt_builder: PromptBuilder,
    dialog: Dialog,
) -> str | None:
    """
    proceseaza o intrebare deja transcrisa

    Flow-ul e asa:
        text
        -> detectare laborator
        -> Retriever
        -> PromptBuilder
        -> Gemini
        -> salvare in Conversation
        -> returnare raspuns pentru XTTS
    """

    cleaned_transcription = transcription.strip()

    if not cleaned_transcription:
        print("\nWhisper did not detect any usable speech.")
        return None

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
        language=response_language,
    )

    print("\nSending prompt to Gemini...")

    try:
        answer = dialog.generate_response(
            prompt
        )

    except RuntimeError as error:
        print("\nGemini request failed:")
        print(error)
        return None

    if not answer or not answer.strip():
        print(
            "\nGemini did not return a usable answer."
        )
        return None

    cleaned_answer = answer.strip()

    conversation.add_user_message(
        cleaned_transcription
    )

    conversation.add_assistant_message(
        cleaned_answer
    )

    print("\nTIAGo response:")
    print(cleaned_answer)

    return cleaned_answer


def main() -> None:
    args = parse_arguments()

    if args.duration <= 0:
        raise ValueError(
            "Recording duration must be greater than zero."
        )

    if args.top_k <= 0:
        raise ValueError(
            "Top-k must be greater than zero."
        )

    recording_path = Path(args.output)
    reference_audio = Path(args.reference_audio)
    output_audio = Path(args.tts_output)

    if not reference_audio.exists():
        raise FileNotFoundError(
            f"Reference audio was not found: "
            f"{reference_audio}"
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

    # XTTS este initializat o singura data.
    # Nu il initializam in while, deoarece modelul s-ar incarca
    # din nou dupa fiecare intrebare.
    tts = XTTS()

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

        answer = process_transcription(
            transcription=transcription,
            response_language=args.response_language,
            top_k=args.top_k,
            context_selector=context_selector,
            conversation=conversation,
            retriever=retriever,
            prompt_builder=prompt_builder,
            dialog=dialog,
        )

        # Daca Whisper sau Gemini nu au produs un rezultat,
        # nu avem ce text sa trimitem catre XTTS.
        if answer is None:
            print()
            continue

        print("\nGenerating speech with XTTS...")

        try:
            generated_audio = tts.synthesize(
                text=answer,
                reference_audio=reference_audio,
                output_audio=output_audio,
                language="en",
            )

        except (
            RuntimeError,
            ValueError,
            FileNotFoundError,
        ) as error:
            print("\nXTTS generation failed:")
            print(error)
            print()
            continue

        # Este folosita mereu aceeasi cale.
        # La fiecare raspuns nou, answer.wav este suprascris.
        print(
            "\nGenerated response saved to:"
        )
        print(generated_audio)

        print()


if __name__ == "__main__":
    main()