from pathlib import Path
import wave
from piper import PiperVoice
from piper.config import SynthesisConfig


class RomanianTTS:
    def __init__(
        self,
        model_path: str | Path,
        trailing_silence_ms: int = 250,
    ) -> None:
        print("Loading Romanian Piper model...")

        self.voice = PiperVoice.load(str(model_path))

        self.synthesis_config = SynthesisConfig(
            length_scale=1.1,
            noise_scale=0.5,
            noise_w_scale=0.6,
        )

        self.trailing_silence_ms = trailing_silence_ms

    def synthesize(self, text: str, output_audio: str | Path) -> Path:
        output_path = Path(output_audio)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cleaned_text = text.strip()

        with wave.open(str(output_path), "wb") as wav_file:
            self.voice.synthesize_wav(
                cleaned_text,
                wav_file,
                syn_config=self.synthesis_config,
            )

        # fara asta piper taie ultima silaba destul de des, deci adaugam putina liniste la coada
        self._add_trailing_silence(output_path)

        return output_path

    def _add_trailing_silence(self, audio_path: Path) -> None:
        """
        Adaugă puțină liniște la finalul fișierului WAV,
        pentru a evita tăierea ultimei silabe.
        """
        # citim tot fisierul ca sa aflam parametrii audio si frame-urile existente
        with wave.open(str(audio_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())

        silence_frames_count = int(sample_rate * self.trailing_silence_ms / 1000)
        silence = b"\x00" * (silence_frames_count * channels * sample_width)

        # rescriem fisierul cu frame-urile vechi plus liniste la final
        with wave.open(str(audio_path), "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(frames + silence)
