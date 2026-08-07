import socket
import struct
from pathlib import Path
from typing import Union


HEADER_SIZE = 8
BUFFER_SIZE = 64 * 1024


# ==========================
# SERVER -> CLIENT
# ==========================

MESSAGE_ERROR = b"\x00"
MESSAGE_AUDIO_CHUNK = b"\x01"
MESSAGE_END_RESPONSE = b"\x02"
MESSAGE_STANDBY = b"\x03"


# ==========================
# CLIENT -> SERVER
# ==========================

MESSAGE_START_SESSION = b"\x10"
MESSAGE_AUDIO_REQUEST = b"\x11"



def receive_exactly(connection: socket.socket,number_of_bytes: int,)->bytes:
    """
    primeste exact numarul de bytes cerut
    """

    received_data = bytearray()


    while len(received_data) < number_of_bytes:

        chunk = connection.recv(
            number_of_bytes - len(received_data)
        )


        if not chunk:
            raise ConnectionError("Connection closed before all data was received.")

        received_data.extend(
            chunk
        )
    return bytes(received_data)



def send_message_type(connection: socket.socket,message_type: bytes,)->None:
    """
    trimite mesaj de control
    """

    connection.sendall(message_type)


def receive_message_type(connection: socket.socket,)->bytes:
    """
    primeste tipul mesajului
    """

    message_type = receive_exactly(connection,1,)

    valid_messages = (
        MESSAGE_ERROR,
        MESSAGE_AUDIO_CHUNK,
        MESSAGE_END_RESPONSE,
        MESSAGE_STANDBY,
        MESSAGE_START_SESSION,
        MESSAGE_AUDIO_REQUEST,
    )


    if message_type not in valid_messages:
        raise ConnectionError("Unknown message type received.")
    return message_type


def send_start_session(connection: socket.socket,)->None:
    """
    Client -> Server
    porneste o conversatie Gemini noua.
    """

    send_message_type(connection,MESSAGE_START_SESSION,)


def send_audio_request(connection: socket.socket,)->None:
    """
    Client -> Server

    Anunta ca urmeaza WAV-ul
    """

    send_message_type(connection,MESSAGE_AUDIO_REQUEST,)


def send_file(
    connection: socket.socket,
    file_path: Union[str, Path],
) -> None:
    """
    Trimite un fisier:

    1. dimensiune 8 bytes
    2. continut
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            "File not found: {}".format(
                path
            )
        )


    file_size = path.stat().st_size
    connection.sendall(
        struct.pack("!Q",file_size,)
    )

    with path.open("rb") as file:
        while True:
            chunk = file.read(BUFFER_SIZE)
            if not chunk:
                break
            connection.sendall(chunk)


def receive_file(
    connection: socket.socket,
    output_path: Union[str, Path],
) -> Path:
    """
    primeste un fisier
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True,exist_ok=True,)

    header = receive_exactly(connection,HEADER_SIZE,)
    file_size = struct.unpack("!Q",header,)[0]


    remaining_bytes = file_size

    with path.open("wb") as file:
        while remaining_bytes > 0:
            chunk = connection.recv(
                min(BUFFER_SIZE,remaining_bytes,)
            )
            if not chunk:
                raise ConnectionError("Connection closed while receiving file.")
                
            file.write(chunk)
            remaining_bytes -= len(chunk)
            
    return path

def send_text(connection: socket.socket,message: str,)->None:

    encoded_message = message.encode("utf-8")

    connection.sendall(
        struct.pack("!Q",len(encoded_message),)
    )
    
    connection.sendall(encoded_message)
    
def receive_text(connection: socket.socket,)->str:

    header = receive_exactly(connection,HEADER_SIZE,)
    message_size = struct.unpack("!Q",header,)[0]

    message = receive_exactly(connection,message_size,)
    
    return message.decode("utf-8")
