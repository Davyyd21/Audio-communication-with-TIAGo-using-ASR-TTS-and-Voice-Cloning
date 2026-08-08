from enum import Enum


class SessionState(Enum):
    """
    starile posibile ale conversatiei tiago.
    """
    ACTIVE = "active"
    STANDBY = "standby"


class SessionManager:
    """
    gestioneaza starea sesiunii conversationale tiago.

    active:
        robotul asculta, proceseaza intrebari si raspunde.
    standby:
        robotul este oprit temporar si asteapta
        pornirea unei conversatii noi.

    responsabilitati:
    - schimbarea starii conversatiei;
    - detectarea comenzilor de oprire;
    - detectarea comenzilor de pornire.
    """

    def __init__(self) -> None:
        # la pornire, robotul e mereu activ
        self.state = SessionState.ACTIVE

    def enter_standby(self) -> None:
        """
        pune tiago in modul standby.
        """
        self.state = SessionState.STANDBY
        print("TIAGo entered standby mode.")

    def start_new_session(self) -> None:
        """
        porneste o conversatie noua.
        """
        self.state = SessionState.ACTIVE
        print("TIAGo started a new conversation.")

    def is_active(self) -> bool:
        """
        verifica daca robotul este activ.
        """
        return self.state == SessionState.ACTIVE

    def is_standby(self) -> bool:
        """
        verifica daca robotul este in standby.
        """
        return self.state == SessionState.STANDBY

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        normalizeaza textul primit de la whisper.

        whisper poate returna:
        - diacritice;
        - variatii de scriere;
        - litere mari/mici.
        """
        # transforma in litere mici, elimina spatii goale si inlocuieste diacriticele cu litere simple
        return (
            text.lower()
            .strip()
            .replace("ă", "a")
            .replace("â", "a")
            .replace("î", "i")
            .replace("ș", "s")
            .replace("ş", "s")
            .replace("ț", "t")
            .replace("ţ", "t")
        )

    def is_stop_command(self, text: str) -> bool:
        """
        detecteaza comenzi vocale pentru standby.

        exemple acceptate:
        - tiago stop
        - tiago opreste
        - stop tiago
        - opreste-te tiago
        """
        # normalizeaza textul primit ca sa poata fi comparat corect
        normalized = self.normalize_text(text)

        # lista de comenzi acceptate pentru intrarea in standby
        stop_commands = [
            "tiago stop",
            "stop tiago",
            "tiago opreste",
            "opreste tiago",
            "tiago opreste-te",
            "opreste-te tiago",
            "intra in standby",
            "treci in standby",
        ]

        # verifica daca vreo comanda din lista se regaseste in textul normalizat
        return any(command in normalized for command in stop_commands)

    def is_start_command(self, text: str) -> bool:
        """
        detecteaza comenzi pentru revenirea din standby.

        exemple:
        - tiago porneste
        - incepe conversatia
        - revino
        """
        # normalizeaza textul primit ca sa poata fi comparat corect
        normalized = self.normalize_text(text)

        # lista de comenzi acceptate pentru pornirea unei conversatii noi
        start_commands = [
            "tiago porneste",
            "porneste tiago",
            "tiago revino",
            "revino tiago",
            "incepe conversatia",
            "porneste conversatia",
            "start conversatie",
        ]

        # verifica daca vreo comanda din lista se regaseste in textul normalizat
        return any(command in normalized for command in start_commands)
