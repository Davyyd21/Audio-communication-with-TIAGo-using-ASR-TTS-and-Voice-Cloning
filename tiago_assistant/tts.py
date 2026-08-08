from pathlib import Path
import wave
from piper import PiperVoice
from piper.config import SynthesisConfig


class RomanianTTS:
    def __init__(self, model_path: str | Path, trailing_silence_ms: int = 250) -> None:
        print("Loading Romanian Piper model...")

        # incarca modelul piper de la path-ul primit
        self.voice = PiperVoice.load(str(model_path))

        # seteaza parametrii de sinteza (viteza, zgomot etc)
        self.synthesis_config = SynthesisConfig(
            length_scale=1.1,
            noise_scale=0.5,
            noise_w_scale=0.6,
        )

        self.trailing_silence_ms = trailing_silence_ms

    def synthesize(self, text: str, output_audio: str | Path) -> Path:
        output_path = Path(output_audio)

        # creeaza folderele necesare daca nu exista deja
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # elimina spatiile goale de la inceput/sfarsit
        cleaned_text = text.strip()

        # deschide fisierul wav in mod scriere si genereaza audio-ul din text
        with wave.open(str(output_path), "wb") as wav_file:
            self.voice.synthesize_wav(cleaned_text, wav_file, syn_config=self.synthesis_config)

        # adauga putina liniste la final ca sa nu se taie ultima silaba
        self._add_trailing_silence(output_path)

        return output_path

    def _add_trailing_silence(self, audio_path: Path) -> None:
        """
        adauga putina liniste la finalul fisierului wav,
        pentru a evita taierea ultimei silabe.
        """
        # citeste fisierul wav existent si extrage informatiile despre el
        with wave.open(str(audio_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())

        # calculeaza cate frame-uri de liniste sunt necesare pentru durata dorita
        silence_frames_count = int(sample_rate * self.trailing_silence_ms / 1000)

        # creeaza bytes-ii de liniste (zero-uri), tinand cont de canale si latimea sample-ului
        silence = b"\x00" * (silence_frames_count * channels * sample_width)

        # rescrie fisierul wav cu frame-urile originale plus linistea adaugata la final
        with wave.open(str(audio_path), "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(frames + silence)
