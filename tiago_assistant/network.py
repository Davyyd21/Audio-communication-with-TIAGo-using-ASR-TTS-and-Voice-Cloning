import socket
import struct
from pathlib import Path
from typing import Union


HEADER_SIZE = 8
BUFFER_SIZE = 64 * 1024


# ==================================================
# SERVER -> CLIENT
# ==================================================

MESSAGE_ERROR = b"\x00"
MESSAGE_AUDIO_CHUNK = b"\x01"
MESSAGE_END_RESPONSE = b"\x02"
MESSAGE_STANDBY = b"\x03"


# ==================================================
# CLIENT -> SERVER
# ==================================================

MESSAGE_START_SESSION = b"\x10"
MESSAGE_AUDIO_REQUEST = b"\x11"



def receive_exactly(
    connection: socket.socket,
    number_of_bytes: int,
) -> bytes:
    """
    Primește exact numărul de bytes cerut.
    """

    received_data = bytearray()


    while len(received_data) < number_of_bytes:

        chunk = connection.recv(
            number_of_bytes - len(received_data)
        )


        if not chunk:

            raise ConnectionError(
                "Connection closed before receiving all data."
            )


        received_data.extend(
            chunk
        )


    return bytes(received_data)



# ==================================================
# MESSAGE TYPES
# ==================================================


def send_message_type(
    connection: socket.socket,
    message_type: bytes,
) -> None:
    """
    Trimite un mesaj de control de 1 byte.
    """

    connection.sendall(
        message_type
    )



def receive_message_type(
    connection: socket.socket,
) -> bytes:

    message_type = receive_exactly(
        connection,
        1,
    )


    valid_types = (
        MESSAGE_ERROR,
        MESSAGE_AUDIO_CHUNK,
        MESSAGE_END_RESPONSE,
        MESSAGE_STANDBY,
        MESSAGE_START_SESSION,
        MESSAGE_AUDIO_REQUEST,
    )


    if message_type not in valid_types:

        raise ConnectionError(
            "Unknown message type: {}".format(
                message_type
            )
        )


    return message_type


# ==================================================
# SESSION CONTROL
# ==================================================


def send_start_session(
    connection: socket.socket,
) -> None:
    """
    Client -> Server

    Cere o conversație nouă.
    """

    send_message_type(
        connection,
        MESSAGE_START_SESSION,
    )



def send_audio_request(
    connection: socket.socket,
) -> None:
    """
    Client -> Server

    Anunță că urmează trimiterea unui WAV.
    """

    send_message_type(
        connection,
        MESSAGE_AUDIO_REQUEST,
    )



def send_standby(
    connection: socket.socket,
) -> None:
    """
    Server -> Client

    Pune TIAGo în standby.
    """

    send_message_type(
        connection,
        MESSAGE_STANDBY,
    )



# ==================================================
# FILE TRANSFER
# ==================================================


def send_file(
    connection: socket.socket,
    file_path: Union[str, Path],
) -> None:
    """
    Trimite un fișier:

    1. dimensiune fișier (8 bytes)
    2. conținut fișier
    """

    path = Path(
        file_path
    )


    if not path.exists():

        raise FileNotFoundError(
            "File not found: {}".format(
                path
            )
        )


    if not path.is_file():

        raise ValueError(
            "Path is not a file: {}".format(
                path
            )
        )


    file_size = path.stat().st_size


    connection.sendall(
        struct.pack(
            "!Q",
            file_size,
        )
    )


    with path.open(
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                BUFFER_SIZE
            )


            if not chunk:

                break


            connection.sendall(
                chunk
            )



def receive_file(
    connection: socket.socket,
    output_path: Union[str, Path],
) -> Path:
    """
    Primește un fișier.
    """

    path = Path(
        output_path
    )


    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    header = receive_exactly(
        connection,
        HEADER_SIZE,
    )


    file_size = struct.unpack(
        "!Q",
        header,
    )[0]


    remaining_bytes = file_size


    with path.open(
        "wb"
    ) as file:

        while remaining_bytes > 0:

            chunk = connection.recv(
                min(
                    BUFFER_SIZE,
                    remaining_bytes,
                )
            )


            if not chunk:

                raise ConnectionError(
                    "Connection closed while receiving file."
                )


            file.write(
                chunk
            )


            remaining_bytes -= len(chunk)


    return path



# ==================================================
# SERVER AUDIO RESPONSE
# ==================================================


def send_audio_chunk(
    connection: socket.socket,
    audio_path: Union[str, Path],
) -> None:
    """
    Trimite un fragment audio către client.
    """

    send_message_type(
        connection,
        MESSAGE_AUDIO_CHUNK,
    )


    send_file(
        connection,
        audio_path,
    )



def send_end_response(
    connection: socket.socket,
) -> None:
    """
    Anunță clientul că răspunsul s-a terminat.
    """

    send_message_type(
        connection,
        MESSAGE_END_RESPONSE,
    )



# ==================================================
# ERROR HANDLING
# ==================================================


def send_error(
    connection: socket.socket,
    message: str,
) -> None:
    """
    Trimite o eroare către client.
    """

    send_message_type(
        connection,
        MESSAGE_ERROR,
    )


    send_text(
        connection,
        message,
    )



# ==================================================
# TEXT TRANSFER
# ==================================================


def send_text(
    connection: socket.socket,
    message: str,
) -> None:
    """
    Trimite text cu dimensiune prefixată.
    """

    encoded_message = message.encode(
        "utf-8"
    )


    connection.sendall(
        struct.pack(
            "!Q",
            len(encoded_message),
        )
    )


    connection.sendall(
        encoded_message
    )



def receive_text(
    connection: socket.socket,
) -> str:
    """
    Primește text cu dimensiune prefixată.
    """

    header = receive_exactly(
        connection,
        HEADER_SIZE,
    )


    message_size = struct.unpack(
        "!Q",
        header,
    )[0]


    message = receive_exactly(
        connection,
        message_size,
    )


    return message.decode(
        "utf-8"
    )