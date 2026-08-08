from pathlib import Path
import re
import whisper


class ASR:
    """
    Incarca modelul Whisper o singura data si il refoloseste
    pentru a transcrie fisiere audio in limba romana.
    """

    def __init__(self, model_name: str = "base") -> None:
        print(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_path: str | Path, language: str = "ro") -> str:
        """
        transcrie un fisier audio in limba specificata
        """
        # transforma path-ul primit (string sau Path) intr-un obiect Path
        path = Path(audio_path)

        # verifica daca fisierul chiar exista pe disc, altfel opreste executia
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        # apeleaza functia de transcriere din Whisper cu toate setarile necesare
        result = self.model.transcribe(
            str(path),
            language=language,                  # forteaza limba romana
            task="transcribe",                   # transcrie, nu traduce in alta limba
            fp16=False,                           # ruleaza pe CPU (fara acceleratie GPU/fp16)
            condition_on_previous_text=False,     # nu foloseste textul anterior ca sa evite propagarea greselilor
            initial_prompt=(
                "Conversatie in limba romana despre "
                "laboratoare universitare, robotica, "
                "inteligenta artificiala, procesarea semnalelor, "
                "recunoastere vocala, sisteme integrate, "
                "retele inteligente, Tiago, lidar, FPGA "
                "si echipamente de laborator."
            ),                                     # ajuta modelul sa recunoasca termenii tehnici specifici
            temperature=0.0,                      # decodare determinista (rezultat consecvent, fara aleatoriu)
            no_speech_threshold=0.6,              # cat de sigur trebuie sa fie modelul ca nu e vorbire, ca sa ignore segmentul
            logprob_threshold=-1.0,               # filtreaza segmentele in care modelul nu e sigur de rezultat
            compression_ratio_threshold=2.4,      # filtreaza segmentele cu repetitii ciudate/anormale
        )

        # extrage textul din rezultat, daca nu exista foloseste string gol
        text = str(result.get("text", "")).strip()

        # curata textul de simboluri inutile
        text = self._clean_romanian_text(text)

        # daca dupa curatare textul e gol, arunca eroare
        if not text:
            raise ValueError("Whisper returned an empty transcription.")

        # returneaza textul final, curatat
        return text

    @staticmethod
    def _clean_romanian_text(text: str) -> str:
        """
        Elimina simbolurile si caracterele care nu sunt utile
        pentru o transcriere in limba romana.
        Nu corecteaza cuvintele transcrise gresit.
        """
        # pastreaza doar litere (inclusiv diacritice romanesti), cifre, spatii si punctuatie de baza
        cleaned_text = re.sub(r"[^A-Za-zĂÂÎȘȚăâîșț0-9\s.,?!:;\-]", " ", text)

        # inlocuieste orice secventa de spatii multiple cu un singur spatiu
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)

        # elimina spatiile de la inceput si sfarsit
        return cleaned_text.strip()
