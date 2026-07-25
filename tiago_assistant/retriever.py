import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz


@dataclass(frozen=True)
class KnowledgeChunk:
    """
    Reprezintă un fragment încărcat dintr-un fișier din knowledge.
    """

    source: str
    text: str
    position: int


@dataclass(frozen=True)
class SearchResult:
    """
    Reprezintă un rezultat întors de Retriever.
    """

    source: str
    text: str
    score: float
    position: int


class Retriever:
    """
    Încarcă fișierele .txt din directorul knowledge și caută
    fragmente relevante pentru întrebarea utilizatorului.

    Retriever-ul poate:
    - căuta într-un singur laborator;
    - căuta în toate laboratoarele;
    - returna toate fragmentele unui laborator;
    - include fragmentele vecine unui rezultat;
    - formata rezultatele pentru PromptBuilder.
    """

    ROMANIAN_STOP_WORDS = {
        "a",
        "acel",
        "aceasta",
        "acest",
        "acolo",
        "ai",
        "al",
        "ale",
        "am",
        "are",
        "ar",
        "au",
        "ca",
        "care",
        "ce",
        "cel",
        "cele",
        "cu",
        "cum",
        "că",
        "da",
        "dar",
        "de",
        "despre",
        "din",
        "doar",
        "este",
        "fi",
        "fie",
        "in",
        "la",
        "laborator",
        "laboratorul",
        "mai",
        "mi",
        "mult",
        "nu",
        "o",
        "pe",
        "pentru",
        "prin",
        "sa",
        "sau",
        "se",
        "si",
        "spune",
        "sunt",
        "un",
        "una",
        "unei",
        "unui",
        "în",
        "îmi",
        "și",
    }

    GENERAL_PRESENTATION_PATTERNS = (
        "prezinta",
        "prezinta mi",
        "descrie",
        "descrie mi",
        "detalii despre",
        "spune mi despre",
        "ce poti spune despre",
        "ce este laboratorul",
        "informatii despre",
        "prezentare",
        "overview",
    )

    def __init__(
        self,
        knowledge_directory: str | Path,
        minimum_score: float = 0.18,
    ) -> None:
        self.knowledge_directory = Path(
            knowledge_directory
        )

        self.minimum_score = minimum_score

        if not self.knowledge_directory.exists():
            raise FileNotFoundError(
                "Knowledge directory was not found: "
                f"{self.knowledge_directory}"
            )

        if not self.knowledge_directory.is_dir():
            raise NotADirectoryError(
                "Knowledge path is not a directory: "
                f"{self.knowledge_directory}"
            )

        self.chunks = self._load_chunks()

        if not self.chunks:
            raise ValueError(
                "No usable text fragments were found in "
                f"{self.knowledge_directory}"
            )

    def _load_chunks(self) -> list[KnowledgeChunk]:
        """
        Încarcă toate fișierele .txt și le împarte în fragmente.

        Un fragment este, în mod normal, un paragraf delimitat
        printr-un rând gol.
        """

        chunks: list[KnowledgeChunk] = []

        file_paths = sorted(
            self.knowledge_directory.rglob("*.txt")
        )

        for file_path in file_paths:
            try:
                content = file_path.read_text(
                    encoding="utf-8"
                ).strip()
            except UnicodeDecodeError as error:
                raise ValueError(
                    f"Could not read {file_path} as UTF-8."
                ) from error

            if not content:
                continue

            raw_paragraphs = re.split(
                r"\n\s*\n",
                content,
            )

            position = 0

            for raw_paragraph in raw_paragraphs:
                cleaned_paragraph = self._clean_chunk_text(
                    raw_paragraph
                )

                if len(cleaned_paragraph) < 15:
                    continue

                chunks.append(
                    KnowledgeChunk(
                        source=file_path.stem,
                        text=cleaned_paragraph,
                        position=position,
                    )
                )

                position += 1

        return chunks

    @staticmethod
    def _clean_chunk_text(text: str) -> str:
        """
        Elimină spațiile și liniile inutile dintr-un fragment,
        păstrând conținutul într-o formă compactă.
        """

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        return " ".join(lines)

    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalizează un text pentru comparații.

        Exemplu:
            "Învățare Artificială"
        devine:
            "invatare artificiala"
        """

        normalized_text = text.lower().strip()

        normalized_text = unicodedata.normalize(
            "NFD",
            normalized_text,
        )

        normalized_text = "".join(
            character
            for character in normalized_text
            if unicodedata.category(character) != "Mn"
        )

        normalized_text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            normalized_text,
        )

        return " ".join(
            normalized_text.split()
        )

    def _extract_keywords(
        self,
        text: str,
    ) -> set[str]:
        """
        Extrage cuvintele utile pentru compararea întrebării
        cu fragmentele.
        """

        normalized_text = self.normalize(text)

        return {
            word
            for word in normalized_text.split()
            if len(word) >= 3
            and word not in self.ROMANIAN_STOP_WORDS
        }

    def _calculate_score(
        self,
        question: str,
        chunk_text: str,
    ) -> float:
        """
        Calculează scorul dintre întrebare și fragment.

        Scorul combină:
        - proporția cuvintelor-cheie comune;
        - similaritatea fuzzy;
        - similaritatea dintre seturile de tokeni.
        """

        normalized_question = self.normalize(question)
        normalized_chunk = self.normalize(chunk_text)

        question_keywords = self._extract_keywords(
            question
        )

        chunk_keywords = self._extract_keywords(
            chunk_text
        )

        if not normalized_question:
            return 0.0

        if question_keywords:
            common_keywords = (
                question_keywords & chunk_keywords
            )

            keyword_score = (
                len(common_keywords)
                / len(question_keywords)
            )
        else:
            keyword_score = 0.0

        token_set_score = (
            fuzz.token_set_ratio(
                normalized_question,
                normalized_chunk,
            )
            / 100
        )

        partial_score = (
            fuzz.partial_ratio(
                normalized_question,
                normalized_chunk,
            )
            / 100
        )

        final_score = (
            0.55 * keyword_score
            + 0.30 * token_set_score
            + 0.15 * partial_score
        )

        return min(final_score, 1.0)

    def _matches_laboratory(
        self,
        source: str,
        laboratory_name: str,
    ) -> bool:
        """
        Verifică dacă numele sursei corespunde laboratorului.

        Acceptă diferențe precum:
            AIMultimediaLab
            AI Multimedia Lab
        """

        normalized_source = self.normalize(source)
        normalized_laboratory = self.normalize(
            laboratory_name
        )

        compact_source = normalized_source.replace(
            " ",
            "",
        )

        compact_laboratory = (
            normalized_laboratory.replace(
                " ",
                "",
            )
        )

        if compact_source == compact_laboratory:
            return True

        if (
            compact_source in compact_laboratory
            or compact_laboratory in compact_source
        ):
            return True

        similarity = fuzz.ratio(
            compact_source,
            compact_laboratory,
        )

        return similarity >= 72

    def is_general_presentation_question(
        self,
        question: str,
    ) -> bool:
        """
        Detectează întrebările care cer o prezentare generală.

        Pentru acestea este mai util să trimitem toate fragmentele
        laboratorului, nu doar cele mai apropiate lexical.
        """

        normalized_question = self.normalize(
            question
        )

        return any(
            pattern in normalized_question
            for pattern in self.GENERAL_PRESENTATION_PATTERNS
        )

    def search(
        self,
        question: str,
        laboratory_name: str | None = None,
        top_k: int = 3,
        include_neighbors: bool = True,
    ) -> list[SearchResult]:
        """
        Caută fragmente relevante.

        Dacă laboratory_name este oferit, căutarea este limitată
        la laboratorul respectiv.

        include_neighbors=True adaugă fragmentele aflate imediat
        înaintea și după rezultatele principale.
        """

        if not question or not question.strip():
            return []

        if top_k <= 0:
            return []

        scored_results: list[SearchResult] = []

        for chunk in self.chunks:
            if (
                laboratory_name is not None
                and not self._matches_laboratory(
                    chunk.source,
                    laboratory_name,
                )
            ):
                continue

            score = self._calculate_score(
                question,
                chunk.text,
            )

            if score < self.minimum_score:
                continue

            scored_results.append(
                SearchResult(
                    source=chunk.source,
                    text=chunk.text,
                    score=score,
                    position=chunk.position,
                )
            )

        scored_results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        main_results = scored_results[:top_k]

        if not include_neighbors:
            return main_results

        return self._add_neighbor_chunks(
            main_results
        )

    def _add_neighbor_chunks(
        self,
        main_results: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Adaugă fragmentele vecine rezultatelor principale.

        Vecinii primesc un scor puțin mai mic decât fragmentul
        principal pentru a păstra ordinea informațiilor.
        """

        if not main_results:
            return []

        collected_results: dict[
            tuple[str, int],
            SearchResult,
        ] = {}

        for result in main_results:
            main_key = (
                result.source,
                result.position,
            )

            collected_results[main_key] = result

            for neighbor_offset in (-1, 1):
                neighbor_position = (
                    result.position
                    + neighbor_offset
                )

                neighbor = self._find_chunk(
                    source=result.source,
                    position=neighbor_position,
                )

                if neighbor is None:
                    continue

                neighbor_key = (
                    neighbor.source,
                    neighbor.position,
                )

                if neighbor_key in collected_results:
                    continue

                neighbor_score = max(
                    result.score - 0.05,
                    0.0,
                )

                collected_results[neighbor_key] = (
                    SearchResult(
                        source=neighbor.source,
                        text=neighbor.text,
                        score=neighbor_score,
                        position=neighbor.position,
                    )
                )

        ordered_results = list(
            collected_results.values()
        )

        ordered_results.sort(
            key=lambda result: (
                result.source,
                result.position,
            )
        )

        return ordered_results

    def _find_chunk(
        self,
        source: str,
        position: int,
    ) -> KnowledgeChunk | None:
        """
        Găsește un fragment folosind sursa și poziția sa.
        """

        for chunk in self.chunks:
            if (
                chunk.source == source
                and chunk.position == position
            ):
                return chunk

        return None

    def get_laboratory_chunks(
        self,
        laboratory_name: str,
        max_chunks: int | None = None,
    ) -> list[SearchResult]:
        """
        Returnează toate fragmentele laboratorului.

        Este folosit pentru:
        - prezentări generale;
        - întrebări de continuare pentru care căutarea lexicală
          nu găsește rezultate;
        - situații în care utilizatorul spune doar
          „ce echipamente are?”.
        """

        matching_chunks = [
            chunk
            for chunk in self.chunks
            if self._matches_laboratory(
                chunk.source,
                laboratory_name,
            )
        ]

        matching_chunks.sort(
            key=lambda chunk: chunk.position
        )

        if max_chunks is not None:
            matching_chunks = matching_chunks[
                :max_chunks
            ]

        return [
            SearchResult(
                source=chunk.source,
                text=chunk.text,
                score=1.0,
                position=chunk.position,
            )
            for chunk in matching_chunks
        ]

    def has_laboratory(
        self,
        laboratory_name: str,
    ) -> bool:
        """
        Verifică dacă există cel puțin un fișier sau fragment
        pentru laboratorul dat.
        """

        return any(
            self._matches_laboratory(
                chunk.source,
                laboratory_name,
            )
            for chunk in self.chunks
        )

    @staticmethod
    def format_results(
        results: list[SearchResult],
    ) -> str | None:
        """
        Transformă rezultatele într-un context textual
        pentru PromptBuilder.
        """

        if not results:
            return None

        formatted_chunks: list[str] = []

        for index, result in enumerate(
            results,
            start=1,
        ):
            formatted_chunks.append(
                f"[Fragment {index}]\n"
                f"Source: {result.source}\n"
                f"Content: {result.text}"
            )

        return "\n\n".join(
            formatted_chunks
        )