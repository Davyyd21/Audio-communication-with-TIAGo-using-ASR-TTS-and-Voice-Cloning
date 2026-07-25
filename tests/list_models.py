import os

from dotenv import load_dotenv
from google import genai


def main() -> None:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found in the .env file."
        )

    client = genai.Client(api_key=api_key)

    print("\nAvailable Gemini models:\n")

    for model in client.models.list():
        print(model.name)


if __name__ == "__main__":
    main()