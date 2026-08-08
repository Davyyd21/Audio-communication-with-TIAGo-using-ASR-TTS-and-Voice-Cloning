import os
from collections.abc import Iterator
from dotenv import load_dotenv
from google import genai
from tiago_assistant.system_prompt import SYSTEM_PROMPT


class Dialog:
    """
    manager pentru conversatia gemini.
    gemini pastreaza automat istoricul conversatiei
    in cadrul aceleiasi sesiuni.
    system prompt-ul este incarcat o singura data
    la pornirea serverului.
    """

    def __init__(self, model_name: str = "gemini-3-flash-preview", api_key: str | None = None) -> None:
        # incarca variabilele de mediu din fisierul .env
        load_dotenv()
        self.model_name = model_name

        # foloseste api_key-ul primit ca parametru, sau il ia din variabilele de mediu
        resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")

        # daca nu exista api key nicaieri, opreste executia
        if not resolved_api_key:
            raise ValueError("GEMINI_API_KEY was not found.")

        print("Initializing Gemini client...")

        # creeaza clientul Gemini folosind api key-ul gasit
        self.client = genai.Client(api_key=resolved_api_key)

        self.chat = None
        self._create_chat()

        print("Gemini chat session initialized.")

    def _create_chat(self) -> None:
        """
        creeaza sesiunea gemini.
        system prompt-ul este trimis aici,
        nu la fiecare intrebare.
        """
        # creeaza o sesiune noua de chat cu system prompt-ul si temperatura setate
        self.chat = self.client.chats.create(
            model=self.model_name,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "temperature": 0.4,
            },
        )

    def generate_response_stream(self, prompt: str) -> Iterator[str]:
        """
        trimite un mesaj catre gemini
        si returneaza raspunsul pe bucati.
        istoricul este pastrat automat
        de chat session.
        """
        # verifica daca sesiunea de chat a fost initializata
        if self.chat is None:
            raise RuntimeError("Gemini chat session was not initialized.")

        # elimina spatiile goale de la inceput/sfarsit
        cleaned_prompt = prompt.strip()

        # daca promptul e gol dupa curatare, opreste executia
        if not cleaned_prompt:
            raise ValueError("Prompt cannot be empty.")

        try:
            # trimite mesajul si primeste raspunsul sub forma de stream (bucati de text)
            response_stream = self.chat.send_message_stream(cleaned_prompt)

            received_text = False

            # parcurge fiecare bucata de raspuns primita
            for chunk in response_stream:
                # sare peste bucatile fara text
                if not chunk.text:
                    continue

                received_text = True
                # trimite bucata de text mai departe (streaming)
                yield chunk.text

            # daca nu s-a primit deloc text, arunca eroare
            if not received_text:
                raise RuntimeError("Gemini returned an empty response.")

        except Exception as error:
            # prinde orice eroare aparuta si o re-arunca cu mesaj mai clar
            raise RuntimeError("Gemini request failed: {}".format(error)) from error

    # def generate_response(
    #     self,
    #     prompt: str,
    # ) -> str:
    #     """
    #     varianta fara streaming.
    #     """
    #
    #     response_parts = list(
    #         self.generate_response_stream(
    #             prompt
    #         )
    #     )
    #
    #
    #     response = (
    #         "".join(response_parts)
    #         .strip()
    #     )
    #
    #
    #     if not response:
    #
    #         raise RuntimeError(
    #             "Gemini returned an empty response."
    #         )
    #
    #
    #     return response

    def reset_chat(self) -> None:
        """
        creeaza o conversatie noua.
        se foloseste cand incepe un utilizator nou
        sau o demonstratie noua.
        """
        print("Resetting Gemini chat session...")
        # recreeaza sesiunea de chat (istoricul vechi se pierde)
        self._create_chat()
        print("Gemini chat session reset.")
