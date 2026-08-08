import os
import socket
import threading
import time
from pathlib import Path
from typing import Union

import sounddevice as sd
import soundfile as sf

from network import (
    MESSAGE_AUDIO_CHUNK,
    MESSAGE_END_RESPONSE,
    MESSAGE_ERROR,
    MESSAGE_STANDBY,
    receive_file,
    receive_message_type,
    receive_text,
    send_audio_request,
    send_file,
    send_start_session,
)

from vad_recorder import VADRecorder


if os.name == "nt":
    import msvcrt
else:
    import select
    import sys


SERVER_IP = "192.168.1.134"
SERVER_PORT = 5000


QUESTION_AUDIO = Path("samples/question.wav")

ANSWER_CHUNKS_DIRECTORY = Path("samples/answer_chunks")


class ClientState:

    def __init__(self):
        self.running = True
        self.standby = True

        self.lock = threading.Lock()

    def enter_standby(self):
        with self.lock:
            self.standby = True

    def start_conversation(self):
        with self.lock:
            self.standby = False

    def is_standby(self):
        with self.lock:
            return self.standby

    def is_running(self):
        with self.lock:
            return self.running

    def stop(self):
        with self.lock:
            self.running = False


def wait_for_enter(state: ClientState):
    """
    enter:
    - daca este standby -> porneste sesiune noua;
    """
    while state.is_running():
        pressed = False

        # pe windows, verifica daca s-a apasat o tasta folosind msvcrt
        if os.name == "nt":
            if msvcrt.kbhit():
                key = msvcrt.getwch()
                if key == "\r":
                    pressed = True
        else:
            # pe linux/mac, verifica daca stdin are ceva de citit, cu timeout de 0.1s
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)

            if readable:
                sys.stdin.readline()
                pressed = True

        if pressed:
            if state.is_standby():
                print("\nStarting new TIAGo session...")

                try:
                    start_new_session()
                    state.start_conversation()
                    print("Conversation active.")

                except Exception as error:
                    print("Failed starting session:")
                    print(error)

        # pauza mica ca sa nu incarce procesorul degeaba
        time.sleep(0.05)


def play_audio_file(audio_path: Union[str, Path]):
    path = Path(audio_path)

    # citeste fisierul audio de pe disc
    audio, sample_rate = sf.read(str(path), dtype="float32")

    # reda audio-ul si asteapta pana se termina
    sd.play(audio, sample_rate)
    sd.wait()


def clear_old_answer_chunks():
    # creeaza folderul daca nu exista deja
    ANSWER_CHUNKS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # sterge toate fragmentele vechi de raspuns audio
    for file_path in ANSWER_CHUNKS_DIRECTORY.glob("answer_chunk_*.wav"):
        try:
            file_path.unlink()
        except OSError:
            pass


def start_new_session():
    """
    trimite catre server:
    reset gemini + reset laborator.
    """
    # deschide o conexiune noua catre server doar pentru a trimite comanda de start
    with socket.create_connection((SERVER_IP, SERVER_PORT), timeout=30) as connection:
        send_start_session(connection)

    print("New conversation created.")


def send_question_and_play_answer(question_audio: Path, state: ClientState):
    # sterge fragmentele vechi de raspuns inainte de a incepe unul nou
    clear_old_answer_chunks()

    # deschide conexiunea catre server pentru intreaga interactiune (intrebare + raspuns)
    with socket.create_connection((SERVER_IP, SERVER_PORT), timeout=300) as connection:

        # anunta serverul ca urmeaza o intrebare audio
        send_audio_request(connection)

        # trimite fisierul audio cu intrebarea
        send_file(connection, question_audio)

        chunk_number = 0

        while state.is_running():
            # citeste tipul urmatorului mesaj primit de la server
            message_type = receive_message_type(connection)

            if message_type == MESSAGE_ERROR:
                error_message = receive_text(connection)
                raise RuntimeError(error_message)

            if message_type == MESSAGE_STANDBY:
                print("\nTIAGo entered standby.")

                state.enter_standby()
                break

            if message_type == MESSAGE_END_RESPONSE:
                print("\nResponse completed.")
                break

            if message_type == MESSAGE_AUDIO_CHUNK:
                chunk_number += 1

                # construieste numele fisierului pentru fragmentul curent (ex: answer_chunk_001.wav)
                chunk_path = ANSWER_CHUNKS_DIRECTORY / "answer_chunk_{:03d}.wav".format(chunk_number)

                # primeste fragmentul audio de la server si il salveaza pe disc
                received_chunk = receive_file(connection, chunk_path)

                # reda imediat fragmentul primit
                play_audio_file(received_chunk)


def main():
    # creeaza folderele necesare daca nu exista deja
    QUESTION_AUDIO.parent.mkdir(parents=True, exist_ok=True)
    ANSWER_CHUNKS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # initializeaza recorder-ul cu detectie de voce (vad)
    recorder = VADRecorder(
        sample_rate=16000,
        threshold=0.5,
        minimum_speech_duration=0.4,
        silence_duration=1.2,
        maximum_recording_duration=30.0,
        pre_speech_duration=0.3,
    )

    state = ClientState()

    print("TIAGo client started.")
    print("Press Enter to start.")

    # asteapta apasarea de enter inainte de a incepe
    input()

    start_new_session()

    state.start_conversation()

    # porneste un thread separat care asculta continuu tasta enter
    keyboard_thread = threading.Thread(target=wait_for_enter, args=(state,), daemon=True)

    keyboard_thread.start()

    print("Conversation active.")

    try:
        while state.is_running():
            # daca robotul e in standby, nu inregistreaza nimic, doar asteapta
            if state.is_standby():
                time.sleep(0.2)
                continue

            # inregistreaza o intrebare vocala folosind detectia de voce (vad)
            question_path = recorder.record_utterance(
                output_path=QUESTION_AUDIO,
                stop_event=threading.Event(),
            )

            # daca nu s-a inregistrat nimic valid, sare peste iteratia curenta
            if question_path is None:
                continue

            try:
                # trimite intrebarea catre server si reda raspunsul primit
                send_question_and_play_answer(
                    question_audio=question_path,
                    state=state,
                )

            except Exception as error:
                print("Communication error:")
                print(error)
                state.enter_standby()

    except KeyboardInterrupt:
        pass

    finally:
        state.stop()
        sd.stop()
        print("TIAGo client closed.")


if __name__ == "__main__":
    main()
