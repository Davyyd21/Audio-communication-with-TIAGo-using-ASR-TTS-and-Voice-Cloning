import argparse
import socket
from pathlib import Path

from tiago_assistant.asr import ASR
from tiago_assistant.context_selector import ContextSelector
from tiago_assistant.conversation import Conversation
from tiago_assistant.dialog import Dialog
from tiago_assistant.network import (
    receive_file,
    send_error,
    send_file,
    send_success,
)
from tiago_assistant.prompt_builder import PromptBuilder
from tiago_assistant.retriever import Retriever, SearchResult
from tiago_assistant.tts import RomanianTTS


def parse_arguments() -> argparse.Namespace:
    """
    Citeste optiunile primite din terminal.

    Exemplu:
        python main_server.py --port 5000 --model base
    """

    parser = argparse.ArgumentParser(
        description=(
            "Receive microphone audio through Wi-Fi, transcribe it "
            "with Whisper, retrieve laboratory information, generate "
            "a Gemini response, synthesize it with Romanian Piper TTS "
            "and send the generated WAV file back to the client."
        )
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help=(
            "Address on which the server listens. "
            "Default: 0.0.0.0."
        ),
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="TCP port. Default: 5000.",
    )

    parser.add_argument(
        "--model",
        default="base",
        help="Whisper model name. Default: base.",
    )

    parser.add_argument(
        "--input-language",
        default="ro",
        help=(
            "Language spoken by the user and used by Whisper. "
            "Default: ro."
        ),
    )

    parser.add_argument(
        "--response-language",
        default="ro",
        help=(
            "Language requested from Gemini. "
            "Default: ro."
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
        "--received-audio",
        default="samples/input/received_question.wav",
        help=(
            "Path used for the WAV received from the client. "
            "The file is overwritten after every question."
        ),
    )

    parser.add_argument(
        "--tts-model",
        default="models/piper/ro_RO-lili-high.onnx",
        help=(
            "Path to the Romanian Piper ONNX model. "
            "Default: models/piper/ro_RO-lili-high.onnx."
        ),
    )

    parser.add_argument(
        "--answer-audio",
        default="samples/output/answer.wav",
        help=(
            "Path used for the generated Piper response. "
            "The file is overwritten after every answer."
        ),
    )

    return parser.parse_args()


def print_search_results(
    results: list[SearchResult],
) -> None:
    """
    Afiseaza fragmentele selectate de Retriever.
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
    Selecteaza fragmentele care trebuie trimise catre Gemini.

    Logica:

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
    Proceseaza o intrebare deja transcrisa.

    Flow:
        text
        -> detectare laborator
        -> Retriever
        -> PromptBuilder
        -> Gemini
        -> salvare in Conversation
        -> returnare raspuns pentru Piper
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


def process_audio_file(
    audio_path: Path,
    input_language: str,
    response_language: str,
    top_k: int,
    output_audio: Path,
    asr: ASR,
    context_selector: ContextSelector,
    conversation: Conversation,
    retriever: Retriever,
    prompt_builder: PromptBuilder,
    dialog: Dialog,
    tts: RomanianTTS,
) -> Path:
    """
    Ruleaza intregul pipeline pentru un fisier WAV primit.

    Flow:
        question.wav
        -> Whisper
        -> ContextSelector
        -> Retriever
        -> PromptBuilder
        -> Gemini
        -> Romanian Piper TTS
        -> answer.wav
    """

    print("\nTranscribing received audio...")

    transcription = asr.transcribe(
        audio_path=str(audio_path),
        language=input_language,
    )

    answer = process_transcription(
        transcription=transcription,
        response_language=response_language,
        top_k=top_k,
        context_selector=context_selector,
        conversation=conversation,
        retriever=retriever,
        prompt_builder=prompt_builder,
        dialog=dialog,
    )

    if answer is None:
        raise RuntimeError(
            "No usable response could be generated."
        )

    print("\nGenerating speech with Romanian Piper TTS...")

    generated_audio = tts.synthesize(
        text=answer,
        output_audio=output_audio,
    )

    print(
        f"\nGenerated response saved to: "
        f"{generated_audio}"
    )

    return generated_audio


def handle_client(
    connection: socket.socket,
    client_address: tuple[str, int],
    received_audio: Path,
    output_audio: Path,
    args: argparse.Namespace,
    asr: ASR,
    context_selector: ContextSelector,
    conversation: Conversation,
    retriever: Retriever,
    prompt_builder: PromptBuilder,
    dialog: Dialog,
    tts: RomanianTTS,
) -> None:
    """
    Proceseaza o singura intrebare primita de la client.

    Pentru fiecare intrebare:
        primeste question.wav
        -> genereaza answer.wav
        -> trimite answer.wav inapoi
    """

    print(
        f"\nClient connected: "
        f"{client_address[0]}:{client_address[1]}"
    )

    try:
        received_question = receive_file(
            connection,
            received_audio,
        )

        print(
            f"Question received: "
            f"{received_question}"
        )

        generated_answer = process_audio_file(
            audio_path=received_question,
            input_language=args.input_language,
            response_language=args.response_language,
            top_k=args.top_k,
            output_audio=output_audio,
            asr=asr,
            context_selector=context_selector,
            conversation=conversation,
            retriever=retriever,
            prompt_builder=prompt_builder,
            dialog=dialog,
            tts=tts,
        )

        send_success(connection)

        send_file(
            connection,
            generated_answer,
        )

        print(
            "Generated answer sent to client."
        )

    except Exception as error:
        error_message = str(error)

        print("\nRequest processing failed:")
        print(error_message)

        try:
            send_error(
                connection,
                error_message,
            )

        except (
            OSError,
            ConnectionError,
        ):
            print(
                "Could not send the error message to the client."
            )


def main() -> None:
    args = parse_arguments()

    if not 1 <= args.port <= 65535:
        raise ValueError(
            "Port must be between 1 and 65535."
        )

    if args.top_k <= 0:
        raise ValueError(
            "Top-k must be greater than zero."
        )

    received_audio = Path(
        args.received_audio
    )

    tts_model = Path(
        args.tts_model
    )

    tts_config = Path(
        str(tts_model) + ".json"
    )

    output_audio = Path(
        args.answer_audio
    )

    if not tts_model.exists():
        raise FileNotFoundError(
            f"Piper model was not found: "
            f"{tts_model}"
        )

    if not tts_config.exists():
        raise FileNotFoundError(
            f"Piper model configuration was not found: "
            f"{tts_config}"
        )

    received_audio.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_audio.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Initializing TIAGo processing server...")

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

    # Piper este initializat o singura data,
    # inainte de pornirea buclei serverului.
    tts = RomanianTTS(
        model_path=tts_model,
    )

    with socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    ) as server_socket:
        server_socket.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )

        server_socket.bind(
            (
                args.host,
                args.port,
            )
        )

        server_socket.listen(1)

        print("\nTIAGo processing server started.")
        print(
            f"Listening on "
            f"{args.host}:{args.port}"
        )
        print(
            "Waiting for the TIAGo client..."
        )

        while True:
            try:
                connection, client_address = (
                    server_socket.accept()
                )

            except KeyboardInterrupt:
                print("\nProcessing server stopped.")
                break

            with connection:
                connection.settimeout(300)

                handle_client(
                    connection=connection,
                    client_address=client_address,
                    received_audio=received_audio,
                    output_audio=output_audio,
                    args=args,
                    asr=asr,
                    context_selector=context_selector,
                    conversation=conversation,
                    retriever=retriever,
                    prompt_builder=prompt_builder,
                    dialog=dialog,
                    tts=tts,
                )


if __name__ == "__main__":
    main()