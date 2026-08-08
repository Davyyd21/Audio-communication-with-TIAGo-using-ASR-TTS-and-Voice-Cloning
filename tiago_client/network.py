import socket
import struct
from pathlib import Path
from typing import Union


HEADER_SIZE = 8
BUFFER_SIZE = 64 * 1024


# ==========================
# server -> client
# ==========================

MESSAGE_ERROR = b"\x00"
MESSAGE_AUDIO_CHUNK = b"\x01"
MESSAGE_END_RESPONSE = b"\x02"
MESSAGE_STANDBY = b"\x03"


# ==========================
# client -> server
# ==========================

MESSAGE_START_SESSION = b"\x10"
MESSAGE_AUDIO_REQUEST = b"\x11"


def receive_exactly(connection: socket.socket, number_of_bytes: int) -> bytes:
    """
    primeste exact numarul de bytes cerut
    """
    received_data = bytearray()

    # continua sa citeasca pana ajunge la numarul de bytes dorit
    while len(received_data) < number_of_bytes:
        chunk = connection.recv(number_of_bytes - len(received_data))

        # daca nu s-a primit nimic, conexiunea s-a inchis prematur
        if not chunk:
            raise ConnectionError("Connection closed before all data was received.")

        received_data.extend(chunk)

    return bytes(received_data)


def send_message_type(connection: socket.socket, message_type: bytes) -> None:
    """
    trimite mesaj de control
    """
    connection.sendall(message_type)


def receive_message_type(connection: socket.socket) -> bytes:
    """
    primeste tipul mesajului
    """
    # citeste 1 byte, care reprezinta tipul mesajului
    message_type = receive_exactly(connection, 1)

    # lista tuturor tipurilor de mesaje acceptate
    valid_messages = (
        MESSAGE_ERROR,
        MESSAGE_AUDIO_CHUNK,
        MESSAGE_END_RESPONSE,
        MESSAGE_STANDBY,
        MESSAGE_START_SESSION,
        MESSAGE_AUDIO_REQUEST,
    )

    # daca tipul primit nu e in lista, e o eroare
    if message_type not in valid_messages:
        raise ConnectionError("Unknown message type received.")

    return message_type


def send_start_session(connection: socket.socket) -> None:
    """
    client -> server
    porneste o conversatie gemini noua.
    """
    send_message_type(connection, MESSAGE_START_SESSION)


def send_audio_request(connection: socket.socket) -> None:
    """
    client -> server
    anunta ca urmeaza wav-ul
    """
    send_message_type(connection, MESSAGE_AUDIO_REQUEST)


def send_file(connection: socket.socket, file_path: Union[str, Path]) -> None:
    """
    trimite un fisier:
    1. dimensiune 8 bytes
    2. continut
    """
    path = Path(file_path)

    # verifica daca fisierul exista pe disc
    if not path.exists():
        raise FileNotFoundError("File not found: {}".format(path))

    # ia dimensiunea fisierului in bytes
    file_size = path.stat().st_size

    # trimite dimensiunea fisierului pe 8 bytes (unsigned long long, big-endian)
    connection.sendall(struct.pack("!Q", file_size))

    # deschide fisierul in mod citire binara
    with path.open("rb") as file:
        while True:
            # citeste cate un bloc de date de dimensiune buffer_size
            chunk = file.read(BUFFER_SIZE)

            # daca nu mai e nimic de citit, opreste bucla
            if not chunk:
                break

            # trimite blocul de date pe conexiune
            connection.sendall(chunk)


def receive_file(connection: socket.socket, output_path: Union[str, Path]) -> Path:
    """
    primeste un fisier
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


def send_text(connection: socket.socket, message: str) -> None:
    # codifica textul in bytes (utf-8)
    encoded_message = message.encode("utf-8")

    # trimite lungimea textului pe 8 bytes
    connection.sendall(struct.pack("!Q", len(encoded_message)))

    # trimite efectiv textul codificat
    connection.sendall(encoded_message)


def receive_text(connection: socket.socket) -> str:
    # citeste header-ul cu lungimea textului
    header = receive_exactly(connection, HEADER_SIZE)

    # despacheteaza lungimea textului din header
    message_size = struct.unpack("!Q", header)[0]

    # citeste efectiv textul, cate bytes s-a anuntat in header
    message = receive_exactly(connection, message_size)

    # decodifica bytes-ii inapoi in string
    return message.decode("utf-8")
