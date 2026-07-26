class Conversation:
    """
    pastreaza istoricul conversatiei curente si laboratorul activ
    istoricul este pastrat doar în memoria RAM
    cand aplicatia se inchide, istoricul este pierdut
    istoricul vechi este sters ca sa nu se umple foarte mult structura care retine
    ori ce a zis omul ori ce a raspuns chatbot-ul
    """
#also asta e al 4-lea fisier in logica implmentata in program(pastreaza laboratorul activ) dar fisiereul apare si dupa dialog.py pentru salvarea continutului conversatiei(intrebarea omului+raspunsul chat-ului)
    def __init__(self, max_messages: int = 10):
        if max_messages <= 0:
            raise ValueError("max_messages must be greater than zero.")

        self.max_messages = max_messages
        self.messages: list[dict[str, str]] = []
        self.active_laboratory: str | None = None

    def add_user_message(self, content: str) -> None:
        #Adauga în istoric un mesaj al utilizatorului.
        
        self._add_message(role="user",content=content,)

    def add_assistant_message(self, content: str) -> None:
        
        #Adauga în istoric raspunsul asistentului
        self._add_message(role="assistant",content=content,)

    def _add_message(self, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported message role: {role}")

        cleaned_content = content.strip()

        if not cleaned_content:
            raise ValueError("Conversation message cannot be empty.")

        self.messages.append(
            {
                "role": role,
                "content": cleaned_content,
            }
        )

        self._trim_history()

    def _trim_history(self) -> None:
        #Pastreaza doar cele mai recente mesaje.Astfel, promptul nu crește la infinit pe durata conversatiei
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def get_messages(self) -> list[dict[str, str]]:
        #Folosim o copie pentru ca alte module sa nu poata modifica accidental lista interna
        return [message.copy() for message in self.messages]

    def set_active_laboratory(self, laboratory_name: str) -> None:
        #Actualizeaza laboratorul despre care se discuta.
        
        cleaned_name = laboratory_name.strip()

        if not cleaned_name:
            raise ValueError("Active laboratory name cannot be empty.")

        self.active_laboratory = cleaned_name

    def get_active_laboratory(self) -> str | None:
        #Returnează laboratorul activ sau None daca inca nu exista unul.
        return self.active_laboratory

    def clear(self) -> None:
        #Sterge istoricul și laboratorul activ.

        self.messages.clear()
        self.active_laboratory = None