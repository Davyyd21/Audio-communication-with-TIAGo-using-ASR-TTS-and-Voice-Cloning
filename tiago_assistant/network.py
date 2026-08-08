import socket
import struct
from pathlib import Path
from typing import Union


HEADER_SIZE = 8
BUFFER_SIZE = 64 * 1024


# ==================================================
# server -> client
# ==================================================

MESSAGE_ERROR = b"\x00"
MESSAGE_AUDIO_CHUNK = b"\x01"
MESSAGE_END_RESPONSE = b"\x02"
MESSAGE_STANDBY = b"\x03"


# ==================================================
# client -> server
# ==================================================

MESSAGE_START_SESSION = b"\x10"
MESSAGE_AUDIO_REQUEST = b"\x11"


def receive_exactly(connection: socket.socket, number_of_bytes: int) -> bytes:
    """
    primeste exact numarul de bytes cerut.
    """
    received_data = bytearray()

    # continua sa citeasca pana ajunge la numarul de bytes dorit
    while len(received_data) < number_of_bytes:
        # citeste doar cati bytes mai lipsesc
        chunk = connection.recv(number_of_bytes - len(received_data))

        # daca nu s-a primit nimic, conexiunea s-a inchis prematur
        if not chunk:
            raise ConnectionError("Connection closed before receiving all data.")

        received_data.extend(chunk)

    return bytes(received_data)


# ==================================================
# message types
# ==================================================


def send_message_type(connection: socket.socket, message_type: bytes) -> None:
    """
    trimite un mesaj de control de 1 byte.
    """
    connection.sendall(message_type)


def receive_message_type(connection: socket.socket) -> bytes:
    # citeste 1 byte, care reprezinta tipul mesajului
    message_type = receive_exactly(connection, 1)

    # lista tuturor tipurilor de mesaje acceptate
    valid_types = (
        MESSAGE_ERROR,
        MESSAGE_AUDIO_CHUNK,
        MESSAGE_END_RESPONSE,
        MESSAGE_STANDBY,
        MESSAGE_START_SESSION,
        MESSAGE_AUDIO_REQUEST,
    )

    # daca tipul primit nu e in lista, e o eroare
    if message_type not in valid_types:
        raise ConnectionError("Unknown message type: {}".format(message_type))

    return message_type


# ==================================================
# session control
# ==================================================


def send_start_session(connection: socket.socket) -> None:
    """
    client -> server

    cere o conversatie noua.
    """
    send_message_type(connection, MESSAGE_START_SESSION)


def send_audio_request(connection: socket.socket) -> None:
    """
    client -> server

    anunta ca urmeaza trimiterea unui wav.
    """
    send_message_type(connection, MESSAGE_AUDIO_REQUEST)


def send_standby(connection: socket.socket) -> None:
    """
    server -> client

    pune tiago in standby.
    """
    send_message_type(connection, MESSAGE_STANDBY)


# ==================================================
# file transfer
# ==================================================


def send_file(connection: socket.socket, file_path: Union[str, Path]) -> None:
    """
    trimite un fisier:

    1. dimensiune fisier (8 bytes)
    2. continut fisier
    """
    path = Path(file_path)

    # verifica daca fisierul exista pe disc
    if not path.exists():
        raise FileNotFoundError("File not found: {}".format(path))

    # verifica daca path-ul chiar duce catre un fisier (nu director)
    if not path.is_file():
        raise ValueError("Path is not a file: {}".format(path))

    # ia dimensiunea fisierului in bytes
    file_size = path.stat().st_size

    # trimite dimensiunea fisierului pe 8 bytes (unsigned long long, big-endian)
    connection.sendall(struct.pack("!Q", file_size))

    # deschide fisierul in mod citire binara
    with path.open("rb") as file:
        while True:
            # citeste cate un bloc de date de dimensiune BUFFER_SIZE
            chunk = file.read(BUFFER_SIZE)

            # daca nu mai e nimic de citit, opreste bucla
            if not chunk:
                break

            # trimite blocul de date pe conexiune
            connection.sendall(chunk)


def receive_file(connection: socket.socket, output_path: Union[str, Path]) -> Path:
    """
    primeste un fisier.
    """
    path = Path(output_path)

    # creeaza folderele necesare daca nu exista deja
    path.parent.mkdir(parents=True, exist_ok=True)

    # citeste header-ul cu dimensiunea fisierului
    header = receive_exactly(connection, HEADER_SIZE)

    # despacheteaza dimensiunea fisierului din header
    file_size = struct.unpack("!Q", header)[0]

    remaining_bytes = file_size

    # deschide fisierul de destinatie in mod scriere binara
    with path.open("wb") as file:
        while remaining_bytes > 0:
            # citeste cat mai poate, fara sa depaseasca ce mai ramane de primit
            chunk = connection.recv(min(BUFFER_SIZE, remaining_bytes))

            # daca nu s-a primit nimic, conexiunea s-a inchis prematur
            if not chunk:
                raise ConnectionError("Connection closed while receiving file.")

            # scrie blocul primit in fisier
            file.write(chunk)

            # scade din cat mai ramane de primit
            remaining_bytes -= len(chunk)

    return path


# ==================================================
# server audio response
# ==================================================


def send_audio_chunk(connection: socket.socket, audio_path: Union[str, Path]) -> None:
    """
    trimite un fragment audio catre client.
    """
    # anunta tipul de mesaj (chunk audio)
    send_message_type(connection, MESSAGE_AUDIO_CHUNK)

    # trimite efectiv fisierul audio
    send_file(connection, audio_path)


def send_end_response(connection: socket.socket) -> None:
    """
    anunta clientul ca raspunsul s-a terminat.
    """
    send_message_type(connection, MESSAGE_END_RESPONSE)


# ==================================================
# error handling
# ==================================================


def send_error(connection: socket.socket, message: str) -> None:
    """
    trimite o eroare catre client.
    """
    # anunta tipul de mesaj (eroare)
    send_message_type(connection, MESSAGE_ERROR)

    # trimite textul erorii
    send_text(connection, message)


# ==================================================
# text transfer
# ==================================================


def send_text(connection: socket.socket, message: str) -> None:
    """
    trimite text cu dimensiune prefixata.
    """
    # codifica textul in bytes (utf-8)
    encoded_message = message.encode("utf-8")

    # trimite lungimea textului pe 8 bytes
    connection.sendall(struct.pack("!Q", len(encoded_message)))

    # trimite efectiv textul codificat
    connection.sendall(encoded_message)


def receive_text(connection: socket.socket) -> str:
    """
    primeste text cu dimensiune prefixata.
    """
    # citeste header-ul cu lungimea textului
    header = receive_exactly(connection, HEADER_SIZE)

    # despacheteaza lungimea textului din header
    message_size = struct.unpack("!Q", header)[0]

    # citeste efectiv textul, cate bytes s-a anuntat in header
    message = receive_exactly(connection, message_size)

    # decodifica bytes-ii inapoi in string
    return message.decode("utf-8")
