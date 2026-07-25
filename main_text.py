from tiago_assistant.context_selector import (
    ContextSelector,
)
from tiago_assistant.conversation import (
    Conversation,
)
from tiago_assistant.dialog import Dialog
from tiago_assistant.prompt_builder import (
    PromptBuilder,
)
from tiago_assistant.retriever import (
    Retriever,
    SearchResult,
)


EXIT_COMMANDS = {
    "exit",
    "quit",
    "stop",
    "iesire",
    "ieșire",
}

RESET_COMMANDS = {
    "reset",
    "restart",
}

LABORATORY_COMMANDS = {
    "laboratory",
    "laborator",
}

HELP_COMMANDS = {
    "help",
    "ajutor",
}


def print_help() -> None:
    """
    Afișează comenzile disponibile.
    """

    print("\nAvailable commands:")
    print("- exit        closes the application")
    print("- reset       clears the conversation")
    print("- laboratory  shows the active laboratory")
    print("- help         shows this message\n")


def print_search_results(
    results: list[SearchResult],
) -> None:
    """
    Afișează fragmentele selectate de Retriever.

    Această informație este utilă în timpul dezvoltării,
    pentru a putea vedea ce surse ajung la Gemini.
    """

    if not results:
        print(
            "\nNo relevant fragments found."
        )
        return

    print("\nRetrieved fragments:")

    for result in results:
        print(
            f"- {result.source} "
            f"[fragment {result.position + 1}] "
            f"(score: {result.score:.2f})"
        )


def retrieve_context(
    question: str,
    active_laboratory: str | None,
    retriever: Retriever,
) -> list[SearchResult]:
    """
    Decide ce fragmente trebuie folosite pentru întrebare.

    Strategia este:

    1. Dacă întrebarea cere o prezentare generală și există
       un laborator activ, sunt luate toate fragmentele sale.

    2. Pentru o întrebare specifică se caută cele mai relevante
       fragmente din laboratorul activ.

    3. Dacă această căutare nu găsește nimic, sunt returnate
       toate fragmentele laboratorului activ. Acest fallback
       rezolvă întrebările de continuare precum:
           "Ce echipamente are?"
           "Ce domenii studiază?"

    4. Dacă nu există laborator activ, căutarea se face în toate
       fișierele.
    """

    if active_laboratory is not None:
        if (
            retriever
            .is_general_presentation_question(
                question
            )
        ):
            return (
                retriever.get_laboratory_chunks(
                    laboratory_name=(
                        active_laboratory
                    ),
                    max_chunks=10,
                )
            )

        laboratory_results = retriever.search(
            question=question,
            laboratory_name=active_laboratory,
            top_k=3,
            include_neighbors=True,
        )

        if laboratory_results:
            return laboratory_results

        laboratory_fallback = (
            retriever.get_laboratory_chunks(
                laboratory_name=(
                    active_laboratory
                ),
                max_chunks=10,
            )
        )

        if laboratory_fallback:
            return laboratory_fallback

    return retriever.search(
        question=question,
        laboratory_name=None,
        top_k=3,
        include_neighbors=True,
    )


def main() -> None:
    """
    Rulează varianta conversațională text a asistentului.

    Flux:
        întrebare
        -> detectarea laboratorului
        -> Retriever
        -> PromptBuilder
        -> Gemini
        -> salvarea conversației
    """

    context_selector = ContextSelector(
        "knowledge"
    )

    conversation = Conversation(
        max_messages=10
    )

    retriever = Retriever(
        knowledge_directory="knowledge",
        minimum_score=0.18,
    )

    prompt_builder = PromptBuilder()
    dialog = Dialog()

    print("TIAGo assistant started.")
    print(
        "Commands: exit, reset, laboratory, help\n"
    )

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print(
                "\nConversation ended."
            )
            break

        if not question:
            continue

        normalized_command = (
            Retriever.normalize(question)
        )

        if normalized_command in EXIT_COMMANDS:
            print("Conversation ended.")
            break

        if normalized_command in RESET_COMMANDS:
            conversation.clear()

            print(
                "Conversation and active laboratory "
                "were reset.\n"
            )

            continue

        if (
            normalized_command
            in LABORATORY_COMMANDS
        ):
            active_laboratory = (
                conversation
                .get_active_laboratory()
            )

            if active_laboratory is None:
                print(
                    "\nNo laboratory is currently "
                    "active.\n"
                )
            else:
                print(
                    "\nActive laboratory:",
                    active_laboratory,
                    "\n",
                )

            continue

        if normalized_command in HELP_COMMANDS:
            print_help()
            continue

        detected_laboratory, _ = (
            context_selector.get_context(
                question
            )
        )

        if detected_laboratory is not None:
            conversation.set_active_laboratory(
                detected_laboratory
            )

        active_laboratory = (
            conversation
            .get_active_laboratory()
        )

        results = retrieve_context(
            question=question,
            active_laboratory=(
                active_laboratory
            ),
            retriever=retriever,
        )

        print_search_results(results)

        relevant_context = (
            retriever.format_results(
                results
            )
        )

        prompt = prompt_builder.build(
            question=question,
            laboratory_name=(
                active_laboratory
            ),
            laboratory_context=(
                relevant_context
            ),
            conversation_history=(
                conversation.get_messages()
            ),
            language="ro",
        )

        try:
            answer = (
                dialog.generate_response(
                    prompt
                )
            )
        except RuntimeError as error:
            print(
                "\nGemini request failed:"
            )
            print(error)
            print()

            continue

        if not answer or not answer.strip():
            print(
                "\nTIAGo did not return "
                "a usable answer.\n"
            )
            continue

        cleaned_answer = answer.strip()

        conversation.add_user_message(
            question
        )

        conversation.add_assistant_message(
            cleaned_answer
        )

        print(
            f"\nTIAGo: {cleaned_answer}\n"
        )


if __name__ == "__main__":
    main()