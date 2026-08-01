import socket
import struct
from pathlib import Path


HEADER_SIZE = 8
BUFFER_SIZE = 64 * 1024

STATUS_SUCCESS = b"\x01"
STATUS_ERROR = b"\x00"


def receive_exactly(
    connection: socket.socket,
    number_of_bytes: int,
) -> bytes:
    """
    Primeste exact numarul de bytes cerut.

    socket.recv() poate returna doar o parte din date,
    deci apelam recv() pana cand primim tot continutul.
    """

    received_data = bytearray()

    while len(received_data) < number_of_bytes:
        chunk = connection.recv(
            number_of_bytes - len(received_data)
        )

        if not chunk:
            raise ConnectionError(
                "Connection closed before all data was received."
            )

        received_data.extend(chunk)

    return bytes(received_data)


def send_file(
    connection: socket.socket,
    file_path: str | Path,
) -> None:
    """
    Trimite dimensiunea fisierului si apoi continutul.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Path is not a file: {path}"
        )

    file_size = path.stat().st_size

    connection.sendall(
        struct.pack("!Q", file_size)
    )
 #folosim struct.pack pentru a "descifra" mesajul trimis

    with path.open("rb") as file:
        while True:
            chunk = file.read(BUFFER_SIZE)

            if not chunk:
                break

            connection.sendall(chunk)
#citeste constant din path chunkuri care sunt trimise prin sendall

def receive_file(
    connection: socket.socket,
    output_path: str | Path,
) -> Path:
    """
    Primeste un fisier si il salveaza la output_path.

    Fisierul anterior este suprascris.
    """

    path = Path(output_path)

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

    with path.open("wb") as file:
        while remaining_bytes > 0:
            chunk = connection.recv(
                min(BUFFER_SIZE, remaining_bytes)
            )

            if not chunk:
                raise ConnectionError(
                    "Connection closed while receiving the file."
                )

            file.write(chunk)
            remaining_bytes -= len(chunk)

    return path


def send_text(
    connection: socket.socket,
    message: str,
) -> None:
    """
    Trimite un mesaj text precedat de dimensiunea lui.
    """

    encoded_message = message.encode("utf-8")

    connection.sendall(
        struct.pack("!Q", len(encoded_message))
    )

    connection.sendall(encoded_message)


def receive_text(
    connection: socket.socket,
) -> str:
    """
    Primeste un mesaj text precedat de dimensiunea lui.
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

    return message.decode("utf-8")


def send_success(
    connection: socket.socket,
) -> None:
    """
    Anunta clientul ca raspunsul a fost generat.
    """

    connection.sendall(STATUS_SUCCESS)


def send_error(
    connection: socket.socket,
    message: str,
) -> None:
    """
    Anunta clientul ca procesarea a esuat.
    """

    connection.sendall(STATUS_ERROR)
    send_text(connection, message)


def receive_status(
    connection: socket.socket,
) -> tuple[bool, str | None]:
    """
    Primeste starea procesarii.
    """

    status = receive_exactly(
        connection,
        1,
    )

    if status == STATUS_SUCCESS:
        return True, None

    if status == STATUS_ERROR:
        return False, receive_text(connection)

    raise ConnectionError(
        "Unknown response status."
    )