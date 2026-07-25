import os

from dotenv import load_dotenv
from google import genai


class Dialog:
    #trimite promptul catre Gemini și returneaza raspunsul text
    #clasa nu construieste promptul si nu gestioneaza istoricul
    #responsabilitatea ei este doar comunicarea cu modelul de limbaj,nothing more nothing less

    def __init__(self,model_name: str = "gemini-3-flash-preview",api_key: str | None = None,):

        load_dotenv()

        self.model_name = model_name

        resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not resolved_api_key:
            raise ValueError(
                "GEMINI_API_KEY was not found. "
                "Add it to the .env file."
            )

        self.client = genai.Client(
            api_key=resolved_api_key
        )

    def generate_response(self, prompt: str) -> str:

        cleaned_prompt = prompt.strip()

        if not cleaned_prompt:
            raise ValueError(
                "The prompt sent to Gemini cannot be empty."
            )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=cleaned_prompt,
            )
        except Exception as error:
            raise RuntimeError(
                f"Gemini API request failed: {error}"
            ) from error

        if not response.text:
            raise ValueError(
                "Gemini returned an empty text response."
            )

        return response.text.strip()