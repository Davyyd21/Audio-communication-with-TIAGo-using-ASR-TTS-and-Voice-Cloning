import os
from collections.abc import Iterator
from dotenv import load_dotenv
from google import genai
from tiago_assistant.system_prompt import SYSTEM_PROMPT


class Dialog:
    """
    Manager pentru conversația Gemini.
    Gemini păstrează automat istoricul conversației
    în cadrul aceleiași sesiuni.
    System prompt-ul este încărcat o singură dată
    la pornirea serverului.
    """

    def __init__(self,model_name: str = "gemini-3-flash-preview",api_key: str | None = None,)->None:
        load_dotenv()

        self.model_name = model_name

        resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not resolved_api_key:
            raise ValueError("GEMINI_API_KEY was not found.")

        print("Initializing Gemini client...")

        self.client = genai.Client(api_key=resolved_api_key)

        self.chat = None
        self._create_chat()

        print("Gemini chat session initialized.")

    def _create_chat(self) -> None:
        """
        Creează sesiunea Gemini.
        System prompt-ul este trimis aici,
        nu la fiecare întrebare.
        """
        self.chat = self.client.chats.create(
            model=self.model_name,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "temperature": 0.4,
            },
        )

    def generate_response_stream(self, prompt: str) -> Iterator[str]:
        """
        Trimite un mesaj către Gemini
        și returnează răspunsul pe bucăți.
        Istoricul este păstrat automat
        de Chat Session.
        """
        if self.chat is None:
            raise RuntimeError("Gemini chat session was not initialized.")

        cleaned_prompt = prompt.strip()

        if not cleaned_prompt:
            raise ValueError("Prompt cannot be empty.")

        try:
            response_stream = self.chat.send_message_stream(cleaned_prompt)

            received_text = False

            # dam yield la fiecare bucata de text pe masura ce vine, nu asteptam tot raspunsul
            for chunk in response_stream:
                if not chunk.text:
                    continue

                received_text = True
                yield chunk.text

            if not received_text:
                raise RuntimeError("Gemini returned an empty response.")

        except Exception as error:
            raise RuntimeError("Gemini request failed: {}".format(error)) from error

    # def generate_response(
    #     self,
    #     prompt: str,
    # ) -> str:
    #     """
    #     Variantă fără streaming.
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
        Creează o conversație nouă.
        Se folosește când începe un utilizator nou
        sau o demonstrație nouă.
        """
        print("Resetting Gemini chat session...")

        self._create_chat()

        print("Gemini chat session reset.")
