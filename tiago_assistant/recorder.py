from pathlib import Path

import sounddevice as sd
import soundfile as sf


class AudioRecorder:
    """
    Înregistrează sunet de la microfon și îl salvează
    într-un fișier WAV.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError(
                "Sample rate must be greater than zero."
            )

        if channels <= 0:
            raise ValueError(
                "Number of channels must be greater than zero."
            )

        self.sample_rate = sample_rate
        self.channels = channels

    def record(
        self,
        output_path: str | Path,
        duration: float,
    ) -> Path:
        """
        Înregistrează sunet pentru numărul de secunde primit
        și salvează rezultatul ca fișier WAV.

        Parametri:
            output_path:
                Calea unde va fi salvat fișierul WAV.

            duration:
                Durata înregistrării, în secunde.

        Returnează:
            Calea fișierului WAV salvat.
        """

        if duration <= 0:
            raise ValueError(
                "Recording duration must be greater than zero."
            )

        resolved_output_path = Path(output_path)

        resolved_output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        number_of_samples = int(
            duration * self.sample_rate
        )

        print(
            f"\nRecording for {duration:.1f} seconds..."
        )

        try:
            recording = sd.rec(
                frames=number_of_samples,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
            )

            sd.wait()

        except Exception as error:
            raise RuntimeError(
                "Audio recording failed. Check the microphone "
                "and the selected Windows input device."
            ) from error

        try:
            sf.write(
                file=str(resolved_output_path),
                data=recording,
                samplerate=self.sample_rate,
                subtype="PCM_16",
            )

        except Exception as error:
            raise RuntimeError(
                f"Could not save the recording to "
                f"{resolved_output_path}."
            ) from error

        print(
            f"Recording saved to: {resolved_output_path}"
        )

        return resolved_output_path