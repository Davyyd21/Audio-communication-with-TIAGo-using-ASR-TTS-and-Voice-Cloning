import os
import socket
import threading
import time
from pathlib import Path
from typing import Union

import sounddevice as sd
import soundfile as sf

from network import (
    receive_file,
    receive_status,
    send_file,
)
from vad_recorder import VADRecorder



if os.name == "nt":
    import msvcrt
else:
    import select
    import sys


SERVER_IP = "10.41.192.83"
SERVER_PORT = 5000

QUESTION_AUDIO = Path("samples/question.wav")
ANSWER_AUDIO = Path("samples/answer.wav")


def wait_for_stop_enter(
    stop_event: threading.Event,
) -> None:
    """
    Asteapta apasarea tastei Enter fara sa blocheze
    bucla principala.

    Windows:
        foloseste msvcrt.

    Linux:
        foloseste select si sys.stdin.
    """

    if os.name == "nt":
        while not stop_event.is_set():
            if msvcrt.kbhit():
                key = msvcrt.getwch()

                if key == "\r":
                    stop_event.set()
                    return

            time.sleep(0.05)

    else:
        while not stop_event.is_set():
            readable_streams, _, _ = select.select(
                [sys.stdin],
                [],
                [],
                0.1,
            )

            if readable_streams:
                sys.stdin.readline()
                stop_event.set()
                return


def play_response(
    audio_path: Union[str, Path],
) -> None:
    """
    Citeste si reda fisierul audio primit de la server.
    """

    path = Path(audio_path)

    if not path.exists():
        raise FileNotFoundError(
            "Answer audio file not found: {}".format(path)
        )

    audio, sample_rate = sf.read(
        str(path),
        dtype="float32",
    )

    sd.play(
        audio,
        sample_rate,
    )

    sd.wait()


def send_question_and_receive_answer(
    question_audio: Path,
    answer_audio: Path,
) -> Path:
    """
    Se conecteaza la serverul de procesare.

    Trimite:
        question.wav

    Primeste:
        statusul procesarii;
        answer.wav, daca procesarea a reusit.
    """

    if not question_audio.exists():
        raise FileNotFoundError(
            "Question audio file not found: {}".format(
                question_audio
            )
        )

    print(
        "Connecting to server {}:{}...".format(
            SERVER_IP,
            SERVER_PORT,
        )
    )

    with socket.create_connection(
        (SERVER_IP, SERVER_PORT),
        timeout=300,
    ) as connection:
        print("Connected to processing laptop.")
        print("Sending question.wav...")

        send_file(
            connection,
            question_audio,
        )

        print("Waiting for processing...")

        success, error_message = receive_status(
            connection
        )

        if not success:
            raise RuntimeError(
                error_message
                or "The processing server returned an error."
            )

        print("Receiving answer.wav...")

        received_answer = receive_file(
            connection,
            answer_audio,
        )

    return received_answer


def main() -> None:
    """
    Porneste clientul conversational.

    Flux:
        VAD
        -> question.wav
        -> server TCP
        -> answer.wav
        -> redare audio
    """

    QUESTION_AUDIO.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ANSWER_AUDIO.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    recorder = VADRecorder(
        sample_rate=16000,
        threshold=0.5,
        minimum_speech_duration=0.4,
        silence_duration=1.2,
        maximum_recording_duration=30.0,
        pre_speech_duration=0.3,
    )

    print("TIAGo conversation client started.")
    print("Server: {}:{}".format(SERVER_IP, SERVER_PORT))
    print("Press Enter to start the conversation.")

    input()

    print("\nConversation mode started.")
    print("A beep means that the client is listening.")
    print("Press Enter again to stop the conversation.")

    stop_event = threading.Event()

    keyboard_thread = threading.Thread(
        target=wait_for_stop_enter,
        args=(stop_event,),
        daemon=True,
    )

    keyboard_thread.start()

    try:
        while not stop_event.is_set():
            try:
                question_path = recorder.record_utterance(
                    output_path=QUESTION_AUDIO,
                    stop_event=stop_event,
                )

            except Exception as error:
                print("\nVAD recording failed:")
                print(error)

                if stop_event.is_set():
                    break

                continue

            if question_path is None:
                if stop_event.is_set():
                    break

                print("No question was recorded.")
                continue

            if stop_event.is_set():
                break

            try:
                answer_path = send_question_and_receive_answer(
                    question_audio=question_path,
                    answer_audio=ANSWER_AUDIO,
                )

            except (
                OSError,
                ConnectionError,
                RuntimeError,
                FileNotFoundError,
            ) as error:
                print("\nCommunication or processing failed:")
                print(error)

                if stop_event.is_set():
                    break

                continue

            if stop_event.is_set():
                break

            print(
                "Answer received: {}".format(
                    answer_path
                )
            )

            print("Playing response...")

            try:
                play_response(
                    answer_path
                )

            except Exception as error:
                print("\nAudio playback failed:")
                print(error)

            if not stop_event.is_set():
                print(
                    "\nResponse finished. "
                    "Returning to listening mode."
                )

    except KeyboardInterrupt:
        print("\nCtrl+C received.")
        stop_event.set()

    finally:
        stop_event.set()
        sd.stop()

        print("\nConversation ended.")


if __name__ == "__main__":
    main()
