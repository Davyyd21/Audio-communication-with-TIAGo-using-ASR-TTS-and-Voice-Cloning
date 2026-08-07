from enum import Enum


class SessionState(Enum):
    """
    Stările posibile ale conversației TIAGo.
    """

    ACTIVE = "active"
    STANDBY = "standby"



class SessionManager:
    """
    Gestionează starea sesiunii conversaționale TIAGo.

    ACTIVE:
        Robotul ascultă, procesează întrebări și răspunde.

    STANDBY:
        Robotul este oprit temporar și așteaptă
        pornirea unei conversații noi.

    Responsabilități:
    - schimbarea stării conversației;
    - detectarea comenzilor de oprire;
    - detectarea comenzilor de pornire.
    """


    def __init__(self) -> None:

        self.state = SessionState.ACTIVE



    def enter_standby(self) -> None:
        """
        Pune TIAGo în modul standby.
        """

        self.state = SessionState.STANDBY

        print(
            "TIAGo entered standby mode."
        )



    def start_new_session(self) -> None:
        """
        Pornește o conversație nouă.
        """

        self.state = SessionState.ACTIVE

        print(
            "TIAGo started a new conversation."
        )



    def is_active(self) -> bool:
        """
        Verifică dacă robotul este activ.
        """

        return (
            self.state == SessionState.ACTIVE
        )



    def is_standby(self) -> bool:
        """
        Verifică dacă robotul este în standby.
        """

        return (
            self.state == SessionState.STANDBY
        )



    @staticmethod
    def normalize_text(
        text: str,
    ) -> str:
        """
        Normalizează textul primit de la Whisper.

        Whisper poate returna:
        - diacritice;
        - variații de scriere;
        - litere mari/mici.
        """

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



    def is_stop_command(
        self,
        text: str,
    ) -> bool:
        """
        Detectează comenzi vocale pentru standby.

        Exemple acceptate:
        - TIAGo stop
        - TIAGo oprește
        - stop TIAGo
        - oprește-te TIAGo
        """

        normalized = (
            self.normalize_text(
                text
            )
        )


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


        return any(
            command in normalized
            for command in stop_commands
        )



    def is_start_command(
        self,
        text: str,
    ) -> bool:
        """
        Detectează comenzi pentru revenirea din standby.

        Exemple:
        - TIAGo pornește
        - începe conversația
        - revino
        """

        normalized = (
            self.normalize_text(
                text
            )
        )


        start_commands = [
            "tiago porneste",
            "porneste tiago",
            "tiago revino",
            "revino tiago",
            "incepe conversatia",
            "porneste conversatia",
            "start conversatie",
        ]


        return any(
            command in normalized
            for command in start_commands
        )