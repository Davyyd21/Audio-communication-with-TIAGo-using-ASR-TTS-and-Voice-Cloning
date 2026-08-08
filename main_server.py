import argparse
import re
import socket
import threading
from pathlib import Path

from tiago_assistant.asr import ASR
from tiago_assistant.context_selector import ContextSelector
from tiago_assistant.dialog import Dialog
from tiago_assistant.laboratory_state import LaboratoryState

from tiago_assistant.network import (
    MESSAGE_AUDIO_REQUEST,
    MESSAGE_START_SESSION,
    receive_file,
    receive_message_type,
    send_audio_chunk,
    send_end_response,
    send_error,
    send_standby,
)

from tiago_assistant.prompt_builder import PromptBuilder

from tiago_assistant.retriever import (
    Retriever,
    SearchResult,
)

from tiago_assistant.tts import RomanianTTS
from tiago_assistant.session_manager import SessionManager


def parse_arguments() -> argparse.Namespace:
    # creeaza parserul de argumente cu descrierea programului
    parser = argparse.ArgumentParser(description=("TIAGo audio communication server."))

    parser.add_argument("--host", default="0.0.0.0")

    parser.add_argument("--port", type=int, default=5000)

    parser.add_argument("--model", default="base")

    parser.add_argument("--input-language", default="ro")

    parser.add_argument("--knowledge", default="knowledge")

    parser.add_argument("--top-k", type=int, default=3)

    parser.add_argument("--received-audio", default="samples/input/received_question.wav")

    parser.add_argument("--tts-model", default="models/piper/ro_RO-lili-high.onnx")

    parser.add_argument("--answer-chunks-directory", default="samples/output/answer_chunks")

    return parser.parse_args()


def print_search_results(results: list[SearchResult]) -> None:
    # daca nu exista rezultate, afiseaza mesaj si iese
    if not results:
        print("No relevant fragments found.")
        return

    print("\nRetrieved fragments:")

    # afiseaza sursa, pozitia si scorul fiecarui fragment gasit
    for result in results:
        print(
            "- {} fragment {} score {:.2f}".format(
                result.source,
                result.position + 1,
                result.score,
            )
        )


def retrieve_context(
    question: str,
    active_laboratory: str | None,
    retriever: Retriever,
    top_k: int,
) -> list[SearchResult]:
    # daca exista un laborator activ, cauta prioritar in contextul lui
    if active_laboratory:

        # daca e o intrebare de tip "prezentare generala", ia toate fragmentele laboratorului
        if retriever.is_general_presentation_question(question):
            return retriever.get_laboratory_chunks(
                laboratory_name=active_laboratory,
                max_chunks=10,
            )

        results = retriever.search(
            question=question,
            laboratory_name=active_laboratory,
            top_k=top_k,
            include_neighbors=True,
        )

        # daca s-a gasit ceva relevant in laboratorul activ, il returneaza
        if results:
            return results

    # altfel, cauta in toate laboratoarele, fara restrictie
    return retriever.search(
        question=question,
        laboratory_name=None,
        top_k=top_k,
        include_neighbors=True,
    )


def build_prompt_for_transcription(
    transcription: str,
    top_k: int,
    context_selector: ContextSelector,
    laboratory_state: LaboratoryState,
    retriever: Retriever,
    prompt_builder: PromptBuilder,
):
    transcription = transcription.strip()

    # daca transcrierea e goala, nu are sens sa continuam
    if not transcription:
        raise RuntimeError("Empty transcription.")

    # incearca sa detecteze despre ce laborator vorbeste utilizatorul
    detected_laboratory, _ = context_selector.get_context(transcription)

    # daca s-a detectat un laborator nou, il seteaza ca activ
    if detected_laboratory:
        laboratory_state.set_active_laboratory(detected_laboratory)

    active_laboratory = laboratory_state.get_active_laboratory()

    # cauta fragmentele relevante pentru intrebare
    results = retrieve_context(
        question=transcription,
        active_laboratory=active_laboratory,
        retriever=retriever,
        top_k=top_k,
    )

    print_search_results(results)

    # formateaza rezultatele intr-un text de context pentru gemini
    context = retriever.format_results(results)

    # construieste promptul final trimis catre gemini
    prompt = prompt_builder.build(
        question=transcription,
        laboratory_name=active_laboratory,
        laboratory_context=context,
    )

    return prompt


def extract_complete_sentences(text_buffer: str):
    sentences = []

    # pattern care prinde propozitii complete, terminate cu . ! sau ?
    pattern = re.compile(r".+?[.!?]+(?=\s|$)", flags=re.DOTALL)

    last_end = 0

    # cauta toate propozitiile complete gasite in text
    for match in pattern.finditer(text_buffer):
        sentence = match.group(0).strip()

        if sentence:
            sentences.append(sentence)

        last_end = match.end()

    # ce a ramas dupa ultima propozitie completa (posibil neterminat inca)
    remaining = text_buffer[last_end:].lstrip()

    return sentences, remaining


def clear_old_audio_chunks(directory: Path):
    # creeaza folderul daca nu exista deja
    directory.mkdir(parents=True, exist_ok=True)

    # sterge toate fragmentele audio vechi
    for file in directory.glob("answer_chunk_*.wav"):
        try:
            file.unlink()
        except OSError:
            pass


def synthesize_and_send_sentence(
    connection: socket.socket,
    sentence: str,
    chunk_number: int,
    chunks_directory: Path,
    tts: RomanianTTS,
):
    # construieste numele fisierului pentru fragmentul curent (ex: answer_chunk_001.wav)
    chunk_path = chunks_directory / "answer_chunk_{:03d}.wav".format(chunk_number)

    # genereaza audio-ul din text folosind tts
    generated = tts.synthesize(text=sentence, output_audio=chunk_path)

    # trimite fragmentul audio catre client
    send_audio_chunk(connection, generated)

    return generated


def stream_gemini_answer_as_audio(
    connection: socket.socket,
    prompt: str,
    dialog: Dialog,
    tts: RomanianTTS,
    chunks_directory: Path,
):
    # sterge fragmentele audio vechi inainte de a incepe un raspuns nou
    clear_old_audio_chunks(chunks_directory)

    text_buffer = ""

    chunk_number = 0

    print("\nGenerating Gemini response...")

    # parcurge raspunsul gemini bucata cu bucata, pe masura ce vine (streaming)
    for response_chunk in dialog.generate_response_stream(prompt):
        text_buffer += response_chunk

        # extrage propozitiile complete deja formate in buffer
        sentences, text_buffer = extract_complete_sentences(text_buffer)

        # pentru fiecare propozitie completa, genereaza audio si trimite-l
        for sentence in sentences:
            chunk_number += 1

            print("TTS chunk {}: {}".format(chunk_number, sentence))

            synthesize_and_send_sentence(
                connection=connection,
                sentence=sentence,
                chunk_number=chunk_number,
                chunks_directory=chunks_directory,
                tts=tts,
            )

    # ce a mai ramas in buffer dupa ce s-a terminat streamingul, se trimite si el
    remaining = text_buffer.strip()

    if remaining:
        chunk_number += 1

        synthesize_and_send_sentence(
            connection=connection,
            sentence=remaining,
            chunk_number=chunk_number,
            chunks_directory=chunks_directory,
            tts=tts,
        )

    # anunta clientul ca raspunsul s-a terminat complet
    send_end_response(connection)


def process_audio_file(
    audio_path: Path,
    connection: socket.socket,
    asr: ASR,
    context_selector: ContextSelector,
    laboratory_state: LaboratoryState,
    retriever: Retriever,
    prompt_builder: PromptBuilder,
    dialog: Dialog,
    tts: RomanianTTS,
    input_language: str,
    top_k: int,
    chunks_directory: Path,
    session_manager: SessionManager,
):
    # daca robotul e in standby, nu proceseaza nimic
    if session_manager.is_standby():
        return

    # transcrie fisierul audio primit in text
    transcription = asr.transcribe(audio_path, language=input_language)

    print("\nTranscription:")
    print(transcription)

    # verifica daca transcrierea e o comanda de oprire
    if session_manager.is_stop_command(transcription):
        print("Stop command detected.")

        dialog.reset_chat()
        laboratory_state.clear()
        session_manager.enter_standby()

        # anunta clientul ca robotul a intrat in standby
        send_standby(connection)

        return

    # construieste promptul pe baza transcrierii si a contextului gasit
    prompt = build_prompt_for_transcription(
        transcription=transcription,
        top_k=top_k,
        context_selector=context_selector,
        laboratory_state=laboratory_state,
        retriever=retriever,
        prompt_builder=prompt_builder,
    )

    # genereaza raspunsul gemini si il trimite catre client sub forma de audio
    stream_gemini_answer_as_audio(
        connection=connection,
        prompt=prompt,
        dialog=dialog,
        tts=tts,
        chunks_directory=chunks_directory,
    )


def handle_client(
    connection: socket.socket,
    address,
    asr: ASR,
    context_selector: ContextSelector,
    laboratory_state: LaboratoryState,
    retriever: Retriever,
    prompt_builder: PromptBuilder,
    dialog: Dialog,
    tts: RomanianTTS,
    received_audio_path: Path,
    input_language: str,
    top_k: int,
    chunks_directory: Path,
    session_manager: SessionManager,
):
    print("Client connected:", address)

    try:
        # citeste tipul mesajului trimis de client
        message_type = receive_message_type(connection)

        if message_type == MESSAGE_START_SESSION:
            print("Creating new Gemini session.")

            dialog.reset_chat()
            laboratory_state.clear()
            session_manager.start_new_session()

            return

        # daca mesajul nu e nici start session, nici audio request, e o eroare
        if message_type != MESSAGE_AUDIO_REQUEST:
            raise RuntimeError("Expected audio request.")

        # primeste fisierul audio trimis de client
        received_file = receive_file(connection, received_audio_path)

        # proceseaza fisierul audio primit (transcriere + raspuns)
        process_audio_file(
            audio_path=received_file,
            connection=connection,
            asr=asr,
            context_selector=context_selector,
            laboratory_state=laboratory_state,
            retriever=retriever,
            prompt_builder=prompt_builder,
            dialog=dialog,
            tts=tts,
            input_language=input_language,
            top_k=top_k,
            chunks_directory=chunks_directory,
            session_manager=session_manager,
        )

    except Exception as error:
        print("Client processing failed:", error)

        # incearca sa trimita eroarea catre client, dar nu opri programul daca nu reuseste
        try:
            send_error(connection, str(error))
        except Exception:
            pass

    finally:
        connection.close()
        print("Client disconnected.")


def main():
    args = parse_arguments()

    received_audio_path = Path(args.received_audio)

    chunks_directory = Path(args.answer_chunks_directory)

    print("Loading components...")

    # incarca toate componentele necesare (asr, retriever, dialog, tts etc)
    asr = ASR(model_name=args.model)

    context_selector = ContextSelector(knowledge_directory=args.knowledge)

    retriever = Retriever(knowledge_directory=args.knowledge)

    prompt_builder = PromptBuilder()

    # gemini session creata la pornirea serverului
    dialog = Dialog()

    tts = RomanianTTS(model_path=args.tts_model)

    laboratory_state = LaboratoryState()

    session_manager = SessionManager()

    # creeaza socket-ul serverului (tcp/ip)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # permite refolosirea rapida a adresei, fara sa astepte eliberarea ei
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server.bind((args.host, args.port))

    # asculta conexiuni, cu o coada de maxim 5 conexiuni in asteptare
    server.listen(5)

    print("Server listening on {}:{}".format(args.host, args.port))

    try:
        while True:
            # accepta o conexiune noua de la un client
            connection, address = server.accept()

            # trateaza fiecare client intr-un thread separat, ca sa nu blocheze serverul
            thread = threading.Thread(
                target=handle_client,
                args=(
                    connection,
                    address,
                    asr,
                    context_selector,
                    laboratory_state,
                    retriever,
                    prompt_builder,
                    dialog,
                    tts,
                    received_audio_path,
                    args.input_language,
                    args.top_k,
                    chunks_directory,
                    session_manager,
                ),
                daemon=True,
            )

            thread.start()

    except KeyboardInterrupt:
        print("Stopping server...")

    finally:
        server.close()


if __name__ == "__main__":
    main()
