from pathlib import Path
import wave

from piper import PiperVoice
from piper.config import SynthesisConfig


class RomanianTTS:
    def __init__(
        self,
        model_path: str | Path,
    ) -> None:

        print("Loading Romanian Piper model...")

        self.voice = PiperVoice.load(
            str(model_path),
        )

        self.synthesis_config = SynthesisConfig(
            length_scale=1.1,
            noise_scale=0.5,
            noise_w_scale=0.6,
        )

    def synthesize(
        self,
        text: str,
        output_audio: str | Path,
    ) -> Path:

        output_path = Path(output_audio)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with wave.open(str(output_path), "wb") as wav_file:
            self.voice.synthesize_wav(
                text,
                wav_file,
                syn_config=self.synthesis_config,
            )

        return output_path