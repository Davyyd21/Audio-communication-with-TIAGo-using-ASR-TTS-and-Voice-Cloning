from pathlib import Path
import re
import unicodedata

from rapidfuzz import fuzz


class ContextSelector:
    LABORATORIES = {
        "SAIL": {
            "file": "SAIL.txt",
            "aliases": [
                "sail",
                "sale",
                "seil",
                "sail lab",
                "sail laboratory",
                "laboratorul sail",
                "laboratorul sale",
            ],
        },
        "SIGMA": {
            "file": "SIGMA.txt",
            "aliases": [
                "sigma",
                "sigmă",
                "sigma lab",
                "sigma laboratory",
                "laboratorul sigma",
            ],
        },
        "AI Multimedia Lab": {
            "file": "AIMultimediaLab.txt",
            "aliases": [
                "ai multimedia",
                "ai multimedia lab",
                "aimultimedia",
                "multimedia lab",
                "laboratorul ai multimedia",
                "laboratorul multimedia",
            ],
        },
        "Robotics Lab": {
            "file": "RoboticsLab.txt",
            "aliases": [
                "robotics",
                "robotics lab",
                "robotic lab",
                "robotica",
                "laboratorul de robotica",
                "laboratorul robotics",
            ],
        },
        "Vision Lab": {
            "file": "VisionLab.txt",
            "aliases": [
                "vision",
                "vision lab",
                "computer vision",
                "laboratorul vision",
                "laboratorul de vedere artificiala",
            ],
        },
        "Cyber Physical Systems Lab": {
            "file": "CyberPhysicalSystemsLab.txt",
            "aliases": [
                "cyber physical systems",
                "cyber physical systems lab",
                "cyberphysical systems",
                "cps lab",
                "laboratorul cyber physical systems",
            ],
        },
        "Intelligent Networks Lab": {
            "file": "IntelligentNetworksLab.txt",
            "aliases": [
                "intelligent networks",
                "intelligent networks lab",
                "network lab",
                "laboratorul intelligent networks",
                "laboratorul de retele inteligente",
            ],
        },
        "Smart Systems Lab": {
            "file": "SmartSystemsLab.txt",
            "aliases": [
                "smart systems",
                "smart systems lab",
                "smart system lab",
                "laboratorul smart systems",
                "laboratorul de sisteme inteligente",
            ],
        },
    }

    def __init__(
        self,
        knowledge_directory: str | Path = "knowledge",
        fuzzy_threshold: int = 80,
    ):
        self.knowledge_directory = Path(knowledge_directory)
        self.fuzzy_threshold = fuzzy_threshold

        if not self.knowledge_directory.exists():
            raise FileNotFoundError(
                f"Knowledge directory not found: {self.knowledge_directory}"
            )

        if not self.knowledge_directory.is_dir():
            raise NotADirectoryError(
                f"Knowledge path is not a directory: {self.knowledge_directory}"
            )

    @staticmethod
    def normalize_text(text: str) -> str:
        #modificam textul nostru ca sa-i fie mai usor de procesat si identificat sintagma dorita
        #practic comparam textul fără să conteze majuscule, diacritice sau punctuație
        text = text.lower().strip()

        text = unicodedata.normalize("NFD", text)
        text = "".join(
            character
            for character in text
            if unicodedata.category(character) != "Mn"#Mark/Nonspacing
        )#am eliminat diacriticele

        text = re.sub(r"[^a-z0-9\s]", " ", text)#tot ce nu se incadreaza in categoriile de caractere mici respectiv cifre sa fie inlocuite cu spatiu gol
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def detect_exact_match(self, question: str) -> str | None:
        normalized_question = self.normalize_text(question)

        for laboratory_name, laboratory_info in self.LABORATORIES.items():
            for alias in laboratory_info["aliases"]:
                normalized_alias = self.normalize_text(alias)

                if normalized_alias in normalized_question:
                    return laboratory_name

        return None

    def detect_fuzzy_match(self, question: str) -> str | None:
        
        #Folosim RapidFuzz dacă nu a fost gasita o potrivire exacta.
        normalized_question = self.normalize_text(question)

        best_laboratory = None
        best_score = 0.0

        for laboratory_name, laboratory_info in self.LABORATORIES.items():
            for alias in laboratory_info["aliases"]:
                normalized_alias = self.normalize_text(alias)

                score = fuzz.partial_ratio(
                    normalized_alias,
                    normalized_question,
                )

                if score > best_score:
                    best_score = score
                    best_laboratory = laboratory_name

        if best_score >= self.fuzzy_threshold: #daca am depasit un scor acolo de similitudine ala e lab-ul cautat
            return best_laboratory

        return None

    def detect_laboratory(self, question: str) -> str | None:
        #ori folosesti prima metoda ori pe a doua cu RapidFuzz
        if not question or not question.strip():
            return None

        laboratory_name = self.detect_exact_match(question)

        if laboratory_name is not None:
            return laboratory_name

        return self.detect_fuzzy_match(question)

    def load_context(self, laboratory_name: str) -> str:
        laboratory_info = self.LABORATORIES.get(laboratory_name)

        if laboratory_info is None:
            raise ValueError(f"Unknown laboratory: {laboratory_name}")

        file_path = (
            self.knowledge_directory
            / laboratory_info["file"]
        )

        if not file_path.exists():
            raise FileNotFoundError(f"Knowledge file not found for "f"{laboratory_name}: {file_path}")

        context = file_path.read_text(encoding="utf-8").strip()

        if not context:
            raise ValueError(
                f"Knowledge file is empty: {file_path}"
            )

        return context  #in caz ca fisierul a fost gasit la path-ul respectiv atunci ii returneaza continutul/descrierea lab-ului
    

    def get_context_by_laboratory(self,laboratory_name: str,)->tuple[str, str]:
        """
        Încarcă informațiile unui laborator deja cunoscut.

        Este folosită pentru întrebările de continuare, când întrebarea
        curentă nu mai conține explicit numele laboratorului.
        """

        context = self.load_context(laboratory_name)

        return laboratory_name, context


    def get_context(self,question: str,) -> tuple[str | None, str | None]:
        laboratory_name = self.detect_laboratory(question)

        if laboratory_name is None:
            return None, None

        context = self.load_context(laboratory_name)

        return laboratory_name, context #returneaza tuplul gasit in caz ca a aparut vreun match