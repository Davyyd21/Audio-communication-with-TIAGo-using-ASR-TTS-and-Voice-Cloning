from tiago_assistant.dialog import Dialog


def main() -> None:
    dialog = Dialog()

    prompt = """
You are a concise voice assistant.
Answer in Romanian.

Question:
What is artificial intelligence?

Answer:
""".strip()

    answer = dialog.generate_response(prompt)

    print("\nGemini response:")
    print(answer)


if __name__ == "__main__":
    main()