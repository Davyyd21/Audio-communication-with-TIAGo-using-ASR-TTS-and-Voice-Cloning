import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz

#asta e al 5-lea program in logica pipeline-ului nostru

'''
Programul e un mini-RAG(Retrieval-Augmented Generation) simplu pentru un chatbot despre laboratoare.
Clasa Retriever incarca fisiere .txt dintr-un folder, le imparte in fragmente (paragrafe curatate).
Pentru o intrebare data, calculeaza un scor de relevanta pentru fiecare fragment
(combinatie de cuvinte cheie comune, fuzzy matching si potrivire partiala),
filtreaza sub un prag minim si returneaza top-K fragmente, plus vecinii lor (paragraful de dinainte/dupa)
ca sa nu piarda context. Poate filtra dupa un laborator anume (potrivire fuzzy pe numele fisierului),
poate detecta intrebari de tip "prezentare generala" si poate returna toate fragmentele unui laborator.
La final formateaza rezultatele intr-un text gata de trimis catre un prompt.Toate astea le facem pentru ca la
inceput imi returna intregul fisier tradus eventual in romana,iar cand l-am segmentat manual si am pus 
cate un rand liber in fisierele txt chatbotul imi zicea propozitia si cand dadea de acel rand liber zicea ca nu
dispune de informatiile necesare,acum doar segmentam fragmentele si le putem lua si pe bucati ca sa trimitem catre prompt
informatia de care are nevoie,ma refer la aia relevanta intrebarii noastre
'''
@dataclass(frozen=True)
class KnowledgeChunk:
    """
    reprezinta o bucata dintr-un fisier din knowledge adica sa nu mai luam toata descrierea lab-ului ci sa il segmentam
    cat mai mult ca sa ofere strict informatiile pe care i le cerem nu sa aiureasca pe acolo sa-mi zica si de echipamente
    cand eu ii cer doar sa-mi zica domeniile de cercetare
    """
    source: str
    text: str
    position: int


@dataclass(frozen=True)#obiectul nu mai poate fi modificat dupa creare
class SearchResult:
    """
    e exact ce intoarce retriever-ul
    """
    source: str
    text: str
    score: float #relevanta fragmentului gasit fata de intrebarea pe care i-am pus-o noi
    position: int


class Retriever:#varianta mai easy de RAG
    #incarca fisierele txt din knowledge si cauta fragmente relevante pentru intrebarea omului

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

    def __init__(self,knowledge_directory: str | Path,minimum_score: float = 0.18,)->None:#ignoram fragmentele cu scor de relevanta mic
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
        #incarcam acolo fisierele txt si le impartim in fragmente,asta oricum am incercat sa o fac prin modul cum am formatat fisierele txt
        
        chunks: list[KnowledgeChunk] = []

        file_paths = sorted(
            self.knowledge_directory.rglob("*.txt")#cauta in sub-tree fisierele
        )

        for file_path in file_paths:
            try:
                content = file_path.read_text(encoding="utf-8").strip()
            except UnicodeDecodeError as error:
                raise ValueError(f"Could not read {file_path} as UTF-8.") from error

            if not content:
                continue

            raw_paragraphs = re.split(
                r"\n\s*\n",
                content,  #impartim fisierele in paragrafe
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
        #elimina spatiile si liniile inutile din fragment
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        return " ".join(lines)

    @staticmethod
    def normalize(text: str) -> str:
        #pregatim textele pentru comparatie, sa n-avem punctuatie,diacritice,litere mari in litere mici
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

        normalized_text = re.sub(r"[^a-z0-9\s]"," ",normalized_text,)

        return " ".join(
            normalized_text.split()
        )

    def _extract_keywords(self,text: str,)->set[str]:
        #extragem doar cuvintele utile pentru a compara intrebarea cu fragmentele extrase din fisierele txt
        #de ex echipamente, SIGMA etc
        normalized_text = self.normalize(text)

        return {
            word
            for word in normalized_text.split()
            if len(word) >= 3
            and word not in self.ROMANIAN_STOP_WORDS
        }

    def _calculate_score(self,question: str,chunk_text: str,)->float:
        #calculam scorul dintre intrebare si fragment---->keyword-uri comune+fuzzy similarity+similaritatea 
        #intre "tokenii" care alcatuiesc cuvintele

        normalized_question = self.normalize(question)
        normalized_chunk = self.normalize(chunk_text)

        question_keywords = self._extract_keywords(question)

        chunk_keywords = self._extract_keywords(chunk_text)

        if not normalized_question:
            return 0.0

        if question_keywords:
            common_keywords = (question_keywords & chunk_keywords)

            keyword_score = (len(common_keywords) / len(question_keywords))
        else:
            keyword_score = 0.0

        token_set_score = (fuzz.token_set_ratio(normalized_question,normalized_chunk,) / 100)

        partial_score = (fuzz.partial_ratio(normalized_question,normalized_chunk,) / 100)
        #verifica daca intrebarea sau o parte asemanatoare apare în textul mai lung
        final_score = (0.55 * keyword_score + 0.30 * token_set_score + 0.15 * partial_score)
        #am dat diverse ponderi de importanta pentru scoruri, cele mai importante sunt cuvintele cheie

        return min(final_score, 1.0)#totusi proportia sa nu depaseasca 1

    def _matches_laboratory(self,source: str,laboratory_name: str,)->bool:
        #prin asta retriever-ul cauta doar in laboratorul activ si verifica daca fisierul respectiv corespunde laboratorului activ

        normalized_source = self.normalize(source)
        normalized_laboratory = self.normalize(laboratory_name)

        compact_source = normalized_source.replace(" ","",)

        compact_laboratory = (normalized_laboratory.replace(" ","",))

        if compact_source == compact_laboratory:
            return True

        if (compact_source in compact_laboratory or compact_laboratory in compact_source):
            return True

        similarity = fuzz.ratio(compact_source,compact_laboratory,)

        return similarity >= 72#am ales un nr random destul de mare ca threshold

    def is_general_presentation_question(self,question: str,)->bool:
        #detecteaza intrebarile care cer o prezentare generala adk nu doar asa pe un fragmentel,ci sa prezinte tot lab-ul
        #trimitem toate fragmentele nu doar cele apropiate "lexical"

        normalized_question = self.normalize(
            question
        )

        return any(
            pattern in normalized_question
            for pattern in self.GENERAL_PRESENTATION_PATTERNS
        )

    def search(self,question: str,laboratory_name: str | None = None,top_k: int = 3,include_neighbors: bool = True,)->list[SearchResult]:
        #cautam doar fragmentele relevante,daca avem numele lab-ului cautarea e limitata doar la fisierul lab-ului
        '''
        pentru fiecare fragment:

        verifica daca apartine laboratorului cerut;
        calculeaza scorul;
        elimina rezultatele sub minimum_score;
        sorteaza rezultatele descrescator;
        pastreaza primele top_k.
        '''
        if not question or not question.strip():
            return []

        if top_k <= 0:
            return []

        scored_results: list[SearchResult] = []

        for chunk in self.chunks:
            if (laboratory_name is not None and not self._matches_laboratory(chunk.source,laboratory_name,)):
                continue

            score = self._calculate_score(question,chunk.text,)

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

    def _add_neighbor_chunks(self,main_results: list[SearchResult],)->list[SearchResult]:
        #poate fragmentele vecine celui gasit contin si ele informatii importante si sa nu rupem informatia
        

        if not main_results:
            return []

        collected_results: dict[tuple[str, int],SearchResult,]={}

        for result in main_results:
            main_key = (
                result.source,
                result.position,
            )

            collected_results[main_key] = result

            for neighbor_offset in (-1, 1):
                neighbor_position = (result.position + neighbor_offset)

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

    def _find_chunk(self,source: str,position: int,)->KnowledgeChunk | None:

        for chunk in self.chunks:
            if (chunk.source == source and chunk.position == position):
                return chunk

        return None

    def get_laboratory_chunks(self,laboratory_name: str,max_chunks: int | None = None,)->list[SearchResult]:
        """
        returneaza toate fragmentele laboratorului->

        este folosit pentru:
        1.prezentari generale
        2.intrebari de continuare pentru care cautarea lexicala
        3.nu gaseste rezultate
        4.situatii in care utilizatorul spune doar „ia zi ce echipamente are?”
        """

        matching_chunks = [chunk for chunk in self.chunks
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

    def has_laboratory(self,laboratory_name: str,)->bool:
        #verificam daca avem un fisier txt pentru laboratorul cautat
        return any(
            self._matches_laboratory(
                chunk.source,
                laboratory_name,
            )
            for chunk in self.chunks
        )

    @staticmethod
    def format_results(results: list[SearchResult],) -> str | None:
        """
        Transformă rezultatele într-un context textual
        pentru PromptBuilder.
        """
        if not results:
            return None

        formatted_chunks: list[str] = []

        for index, result in enumerate(results,start=1,):
            formatted_chunks.append(
                f"[Fragment {index}]\n"
                f"Source: {result.source}\n"
                f"Content: {result.text}"
            )

        return "\n\n".join(
            formatted_chunks
        )