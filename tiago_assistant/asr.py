from pathlib import Path #ca sa lucram cu fisiere si foldere cand e nevoie de accesarea lor

import whisper

#asta e al doilea fisier in pipeline-ul nostru de dezvoltare
class ASR:
    '''
    Clasa ASR e un wrapper peste Whisper facut sa incarci modelul o singura data si sa-l refolosesti.
    In constructor se incarca modelul Whisper (default "base") si se salveaza in self.model,
    ca sa nu se reincarce de fiecare data cand vrei sa transcrii ceva, fiindca incarcarea e greoaie.
    Metoda transcribe primeste calea unui fisier audio si limba (default "ro"),
    transforma calea intr-un obiect Path ca sa poata verifica usor daca fisierul exista,
    iar daca nu exista arunca direct FileNotFoundError ca sa opreasca pipeline-ul din start.
    Daca fisierul exista, il trimite la model cu limba setata pe romana, task-ul setat pe "transcribe"
    (nu translate) si fp16 dezactivat (fiindca fp16 merge doar pe GPU),
    apoi extrage textul din rezultat si elimina spatiile goale de la capete cu .strip().
    La final, daca textul rezultat e gol, arunca ValueError ca sa nu treaca mai departe un string gol
    prin restul procesului, fiindca ar strica pasii urmatori din pipeline.
    '''
    def __init__(self, model_name: str = "base") -> None:#by default am zis sa folosesc modelul 'base' de whisper mai sunt destule tiny sau large da ori e ineficient ori consuma rapid tokeni
        print(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_path: str | Path, language: str = "ro") -> str:
        path = Path(audio_path) #construim in sine obiectul path ca sa nu scriem incontinuu calea catre fisierul audio

        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        result = self.model.transcribe(
            str(path),
            language=language,
            task="transcribe", #whisper-ul trebuie setat ori sa faca transcribe ori sa faca translate
            fp16=False, #aparent fp16 este doar pentru rularea pe GPU
        )

        text = result["text"].strip() #elimina spatiile goale dintr-un sir de caractere asa " wow " -> "wow"

        if not text:
            raise ValueError("Whisper returned an empty transcription.")

        return text