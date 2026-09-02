from pathlib import Path
import re
import torch
import nemo.collections.asr as nemo_asr
from huggingface_hub import hf_hub_download


class ASR:
    """
    SpeD Parakeet Romanian 110M TDT-CTC.
    Modelul este încărcat o singură dată și reutilizat
    pentru toate fișierele audio primite de server.
    """

    MODEL_REPO = "gabrielpirlo/Sped_ParakeetRomanian_110M_TDT-CTC"
    MODEL_FILE = "SpeD-ParakeetRo_110M_TDT-CTC.nemo"

    def __init__(self) -> None:
        print("Loading SpeD Parakeet Romanian 110M TDT-CTC...")

        # descarca modelul din hugging face daca nu exista deja in cache
        model_path = hf_hub_download(
            repo_id=self.MODEL_REPO,
            filename=self.MODEL_FILE,
        )

        # incarca checkpoint-ul .nemo
        self.model = nemo_asr.models.ASRModel.restore_from(restore_path=model_path)

        # tiago/server-ul nostru ruleaza pe cpu
        self.model = self.model.to("cpu")
        self.model.eval()

        print("SpeD Parakeet loaded successfully.")

    def transcribe(self, audio_path: str | Path, language: str = "ro") -> str:
        """
        Transcrie un fișier audio în limba română.
        Păstrăm aceeași interfață ca vechiul Whisper ASR,
        astfel încât restul aplicației să nu trebuiască modificat.
        """
        path = Path(audio_path)

        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        # language a ramas din incercarile anterioare pentru a avea compatibilitate
        with torch.inference_mode():
            output = self.model.transcribe([str(path)], batch_size=1)

        result = output[0]

        # in functie de versiunea nemo, rezultatul poate fi string sau hypothesis
        if isinstance(result, str):
            text = result
        elif hasattr(result, "text"):
            text = result.text
        else:
            text = str(result)

        text = self._clean_romanian_text(text)

        if not text:
            raise ValueError("SpeD Parakeet returned an empty transcription.")

        return text

    @staticmethod
    def _clean_romanian_text(text: str) -> str:
        """
        Curăță caracterele inutile fără să modifice
        cuvintele produse de ASR.
        """
        cleaned_text = re.sub(r"[^A-Za-zĂÂÎȘȚăâîșț0-9\s.,?!:;\-]", " ", text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)

        return cleaned_text.strip()
