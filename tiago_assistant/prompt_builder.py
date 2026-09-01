from typing import Optional


class PromptBuilder:
    """
    Construiește mesajul dinamic trimis către Gemini.

    System prompt-ul conține regulile permanente ale asistentului.
    Acest builder adaugă doar informațiile care se schimbă:
    - laboratorul activ;
    - contextul găsit de Retriever;
    - întrebarea curentă.
    """

    def build(
        self,
        question: str,
        laboratory_name: Optional[str],
        laboratory_context: Optional[str],
    ) -> str:
        """
        Construiește mesajul curent pentru Gemini.

        Parameters:
            question:
                întrebarea utilizatorului.

            laboratory_name:
                laboratorul detectat ca subiect curent.

            laboratory_context:
                fragmente relevante din documentație.
        """

        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError(
                "Question cannot be empty."
            )


        if laboratory_name:
            active_laboratory = laboratory_name
        else:
            active_laboratory = (
                "Nu a fost identificat un laborator activ."
            )


        if laboratory_context:
            context = laboratory_context
        else:
            context = (
                "Nu au fost găsite informații relevante "
                "în documentație."
            )


        return f"""
Laborator activ:
{active_laboratory}

Documentație relevantă:
{context}

Întrebarea utilizatorului:
{cleaned_question}
""".strip()