import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz

# asta e al 5-lea program in logica pipeline-ului nostru

'''
programul e un mini-rag (retrieval-augmented generation) simplu pentru un chatbot despre laboratoare.
clasa retriever incarca fisiere .txt dintr-un folder, le imparte in fragmente (paragrafe curatate).
pentru o intrebare data, calculeaza un scor de relevanta pentru fiecare fragment
(combinatie de cuvinte cheie comune, fuzzy matching si potrivire partiala),
filtreaza sub un prag minim si returneaza top-k fragmente, plus vecinii lor (paragraful de dinainte/dupa)
ca sa nu piarda context. poate filtra dupa un laborator anume (potrivire fuzzy pe numele fisierului),
poate detecta intrebari de tip "prezentare generala" si poate returna toate fragmentele unui laborator.
la final formateaza rezultatele intr-un text gata de trimis catre un prompt. toate astea le facem pentru ca la
inceput imi returna intregul fisier tradus eventual in romana, iar cand l-am segmentat manual si am pus
cate un rand liber in fisierele txt chatbotul imi zicea propozitia si cand dadea de acel rand liber zicea ca nu
dispune de informatiile necesare, acum doar segmentam fragmentele si le putem lua si pe bucati ca sa trimitem catre prompt
informatia de care are nevoie, ma refer la aia relevanta intrebarii noastre
'''


@dataclass(frozen=True)
class KnowledgeChunk:
    """
    reprezinta o bucata dintr-un fisier din knowledge, adica sa nu mai luam toata descrierea lab-ului ci sa il segmentam
    cat mai mult ca sa ofere strict informatiile pe care i le cerem, nu sa aiureasca pe acolo sa-mi zica si de echipamente
    cand eu ii cer doar sa-mi zica domeniile de cercetare
    """
    source: str
    text: str
    position: int


@dataclass(frozen=True)  # obiectul nu mai poate fi modificat dupa creare
class SearchResult:
    """
    e exact ce intoarce retriever-ul
    """
    source: str
    text: str
    score: float  # relevanta fragmentului gasit fata de intrebarea pe care i-am pus-o noi
    position: int


class Retriever:  # varianta mai easy de rag
    # incarca fisierele txt din knowledge si cauta fragmente relevante pentru intrebarea omului

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
        "vorbeste",
        "explica",
    )

    def __init__(self, knowledge_directory: str | Path, minimum_score: float = 0.18) -> None:
        # ignoram fragmentele cu scor de relevanta mic
        self.knowledge_directory = Path(knowledge_directory)

        self.minimum_score = minimum_score

        # verifica daca folderul de knowledge chiar exista
        if not self.knowledge_directory.exists():
            raise FileNotFoundError("Knowledge directory was not found: " f"{self.knowledge_directory}")

        # verifica daca path-ul e chiar un director, nu un fisier
        if not self.knowledge_directory.is_dir():
            raise NotADirectoryError("Knowledge path is not a directory: " f"{self.knowledge_directory}")

        self.chunks = self._load_chunks()

        # daca nu s-a gasit niciun fragment folosibil, opreste executia
        if not self.chunks:
            raise ValueError("No usable text fragments were found in " f"{self.knowledge_directory}")

    def _load_chunks(self) -> list[KnowledgeChunk]:
        # incarcam acolo fisierele txt si le impartim in fragmente, asta oricum am incercat sa o fac prin modul cum am formatat fisierele txt
        chunks: list[KnowledgeChunk] = []

        # cauta in sub-tree fisierele
        file_paths = sorted(self.knowledge_directory.rglob("*.txt"))

        for file_path in file_paths:
            try:
                content = file_path.read_text(encoding="utf-8").strip()
            except UnicodeDecodeError as error:
                raise ValueError(f"Could not read {file_path} as UTF-8.") from error

            # sare peste fisierele goale
            if not content:
                continue

            # impartim fisierele in paragrafe
            raw_paragraphs = re.split(r"\n\s*\n", content)

            position = 0

            for raw_paragraph in raw_paragraphs:
                cleaned_paragraph = self._clean_chunk_text(raw_paragraph)

                # sare peste paragrafele prea scurte ca sa fie utile
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
        # elimina spatiile si liniile inutile din fragment
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        return " ".join(lines)

    @staticmethod
    def normalize(text: str) -> str:
        # pregatim textele pentru comparatie, sa n-avem punctuatie, diacritice, litere mari in litere mici
        normalized_text = text.lower().strip()

        # descompune caracterele cu diacritice in litera de baza + semnul diacritic
        normalized_text = unicodedata.normalize("NFD", normalized_text)

        # elimina semnele diacritice (categoria "Mn" = mark, nonspacing)
        normalized_text = "".join(
            character
            for character in normalized_text
            if unicodedata.category(character) != "Mn"
        )

        # inlocuieste orice caracter care nu e litera/cifra/spatiu cu spatiu
        normalized_text = re.sub(r"[^a-z0-9\s]", " ", normalized_text)

        return " ".join(normalized_text.split())

    def _extract_keywords(self, text: str) -> set[str]:
        # extragem doar cuvintele utile pentru a compara intrebarea cu fragmentele extrase din fisierele txt
        # de ex echipamente, sigma etc
        normalized_text = self.normalize(text)

        return {
            word
            for word in normalized_text.split()
            if len(word) >= 3 and word not in self.ROMANIAN_STOP_WORDS
        }

    def _calculate_score(self, question: str, chunk_text: str) -> float:
        # calculam scorul dintre intrebare si fragment ----> keyword-uri comune + fuzzy similarity + similaritatea
        # intre "tokenii" care alcatuiesc cuvintele
        normalized_question = self.normalize(question)
        normalized_chunk = self.normalize(chunk_text)

        question_keywords = self._extract_keywords(question)

        chunk_keywords = self._extract_keywords(chunk_text)

        # daca intrebarea normalizata e goala, nu are sens sa calculam scor
        if not normalized_question:
            return 0.0

        if question_keywords:
            common_keywords = question_keywords & chunk_keywords

            keyword_score = len(common_keywords) / len(question_keywords)
        else:
            keyword_score = 0.0

        token_set_score = fuzz.token_set_ratio(normalized_question, normalized_chunk) / 100

        # verifica daca intrebarea sau o parte asemanatoare apare in textul mai lung
        partial_score = fuzz.partial_ratio(normalized_question, normalized_chunk) / 100

        # am dat diverse ponderi de importanta pentru scoruri, cele mai importante sunt cuvintele cheie
        final_score = 0.55 * keyword_score + 0.30 * token_set_score + 0.15 * partial_score

        # totusi proportia sa nu depaseasca 1
        return min(final_score, 1.0)

    def _matches_laboratory(self, source: str, laboratory_name: str) -> bool:
        # prin asta retriever-ul cauta doar in laboratorul activ si verifica daca fisierul respectiv corespunde laboratorului activ
        normalized_source = self.normalize(source)
        normalized_laboratory = self.normalize(laboratory_name)

        compact_source = normalized_source.replace(" ", "")

        compact_laboratory = normalized_laboratory.replace(" ", "")

        # potrivire exacta dupa eliminarea spatiilor
        if compact_source == compact_laboratory:
            return True

        # potrivire partiala, unul contine pe celalalt
        if compact_source in compact_laboratory or compact_laboratory in compact_source:
            return True

        similarity = fuzz.ratio(compact_source, compact_laboratory)

        # am ales un nr random destul de mare ca threshold
        return similarity >= 72

    def is_general_presentation_question(self, question: str) -> bool:
        # detecteaza intrebarile care cer o prezentare generala, adica nu doar asa pe un fragment, ci sa prezinte tot lab-ul
        # trimitem toate fragmentele nu doar cele apropiate "lexical"
        normalized_question = self.normalize(question)

        return any(pattern in normalized_question for pattern in self.GENERAL_PRESENTATION_PATTERNS)

    def search(
        self,
        question: str,
        laboratory_name: str | None = None,
        top_k: int = 3,
        include_neighbors: bool = True,
    ) -> list[SearchResult]:
        # cautam doar fragmentele relevante, daca avem numele lab-ului cautarea e limitata doar la fisierul lab-ului
        '''
        pentru fiecare fragment:

        verifica daca apartine laboratorului cerut;
        calculeaza scorul;
        elimina rezultatele sub minimum_score;
        sorteaza rezultatele descrescator;
        pastreaza primele top_k.
        '''
        # daca intrebarea e goala, nu are sens sa cautam
        if not question or not question.strip():
            return []

        # daca top_k e invalid (0 sau negativ), nu returnam nimic
        if top_k <= 0:
            return []

        scored_results: list[SearchResult] = []

        for chunk in self.chunks:
            # sare peste fragmentele care nu apartin laboratorului cerut
            if laboratory_name is not None and not self._matches_laboratory(chunk.source, laboratory_name):
                continue

            score = self._calculate_score(question, chunk.text)

            # sare peste fragmentele cu scor prea mic
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

        # sorteaza descrescator dupa scor (cel mai relevant primul)
        scored_results.sort(key=lambda result: result.score, reverse=True)

        main_results = scored_results[:top_k]

        if not include_neighbors:
            return main_results

        return self._add_neighbor_chunks(main_results)

    def _add_neighbor_chunks(self, main_results: list[SearchResult]) -> list[SearchResult]:
        # poate fragmentele vecine celui gasit contin si ele informatii importante, sa nu rupem informatia
        if not main_results:
            return []

        collected_results: dict[tuple[str, int], SearchResult] = {}

        for result in main_results:
            main_key = (result.source, result.position)

            collected_results[main_key] = result

            # verifica vecinul de dinainte (-1) si cel de dupa (+1)
            for neighbor_offset in (-1, 1):
                neighbor_position = result.position + neighbor_offset

                neighbor = self._find_chunk(source=result.source, position=neighbor_position)

                # daca nu exista vecin la pozitia respectiva, treci mai departe
                if neighbor is None:
                    continue

                neighbor_key = (neighbor.source, neighbor.position)

                # daca vecinul e deja in rezultate, nu-l mai adauga o data
                if neighbor_key in collected_results:
                    continue

                # scorul vecinului e putin mai mic decat cel al rezultatului principal
                neighbor_score = max(result.score - 0.05, 0.0)

                collected_results[neighbor_key] = SearchResult(
                    source=neighbor.source,
                    text=neighbor.text,
                    score=neighbor_score,
                    position=neighbor.position,
                )

        ordered_results = list(collected_results.values())

        # sorteaza rezultatele dupa sursa si pozitie, ca sa fie in ordine logica
        ordered_results.sort(key=lambda result: (result.source, result.position))

        return ordered_results

    def _find_chunk(self, source: str, position: int) -> KnowledgeChunk | None:
        for chunk in self.chunks:
            if chunk.source == source and chunk.position == position:
                return chunk

        return None

    def get_laboratory_chunks(
        self,
        laboratory_name: str,
        max_chunks: int | None = None,
    ) -> list[SearchResult]:
        """
        returneaza toate fragmentele laboratorului ->

        este folosit pentru:
        1. prezentari generale
        2. intrebari de continuare pentru care cautarea lexicala
        3. nu gaseste rezultate
        4. situatii in care utilizatorul spune doar "ia zi ce echipamente are?"
        """
        matching_chunks = [
            chunk
            for chunk in self.chunks
            if self._matches_laboratory(chunk.source, laboratory_name)
        ]

        # sorteaza fragmentele in ordinea lor originala din fisier
        matching_chunks.sort(key=lambda chunk: chunk.position)

        # daca s-a cerut o limita, taie lista la max_chunks
        if max_chunks is not None:
            matching_chunks = matching_chunks[:max_chunks]

        return [
            SearchResult(
                source=chunk.source,
                text=chunk.text,
                score=1.0,
                position=chunk.position,
            )
            for chunk in matching_chunks
        ]

    def has_laboratory(self, laboratory_name: str) -> bool:
        # verificam daca avem un fisier txt pentru laboratorul cautat
        return any(
            self._matches_laboratory(chunk.source, laboratory_name)
            for chunk in self.chunks
        )

    @staticmethod
    def format_results(results: list[SearchResult]) -> str | None:
        """
        transforma rezultatele intr-un context textual
        pentru promptbuilder.
        """
        # daca nu exista rezultate, nu avem ce formata
        if not results:
            return None

        formatted_chunks: list[str] = []

        for index, result in enumerate(results, start=1):
            formatted_chunks.append(
                f"[Fragment {index}]\n"
                f"Source: {result.source}\n"
                f"Content: {result.text}"
            )

        return "\n\n".join(formatted_chunks)
