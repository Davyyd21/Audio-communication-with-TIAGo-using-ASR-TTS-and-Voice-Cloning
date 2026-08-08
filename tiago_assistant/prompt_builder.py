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
        construieste mesajul curent pentru gemini.

        parameters:
            question:
                intrebarea utilizatorului.
            laboratory_name:
                laboratorul detectat ca subiect curent.
            laboratory_context:
                fragmente relevante din documentatie.
        """
        # elimina spatiile goale de la inceput/sfarsit
        cleaned_question = question.strip()

        # daca intrebarea e goala dupa curatare, opreste executia
        if not cleaned_question:
            raise ValueError("Question cannot be empty.")

        # daca exista un laborator detectat, il foloseste, altfel pune un mesaj default
        if laboratory_name:
            active_laboratory = laboratory_name
        else:
            active_laboratory = "Nu a fost identificat un laborator activ."

        # daca exista context relevant, il foloseste, altfel pune un mesaj default
        if laboratory_context:
            context = laboratory_context
        else:
            context = "Nu au fost gasite informatii relevante in documentatie."

        # construieste mesajul final, cu laboratorul activ, contextul si intrebarea
        return f"""
Laborator activ:
{active_laboratory}
Documentație relevantă:
{context}
Întrebarea utilizatorului:
{cleaned_question}
""".strip()
