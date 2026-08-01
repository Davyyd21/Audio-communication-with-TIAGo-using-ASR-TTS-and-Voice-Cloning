import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Union

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from silero_vad import load_silero_vad


class VADRecorder:
    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        minimum_speech_duration: float = 0.4,
        silence_duration: float = 1.2,
        maximum_recording_duration: float = 30.0,
        pre_speech_duration: float = 0.3,
    ) -> None:
        if sample_rate != 16000:
            raise ValueError(
                "Silero VAD expects a sample rate of 16000 Hz."
            )

        if not 0.0 < threshold < 1.0:
            raise ValueError(
                "VAD threshold must be between 0 and 1."
            )

        self.sample_rate = sample_rate
        self.threshold = threshold
        self.chunk_size = 512

        self.minimum_speech_samples = int(
            minimum_speech_duration * sample_rate
        )

        self.silence_samples_required = int(
            silence_duration * sample_rate
        )

        self.maximum_recording_samples = int(
            maximum_recording_duration * sample_rate
        )

        self.pre_speech_chunks = max(
            1,
            int(
                pre_speech_duration
                * sample_rate
                / self.chunk_size
            ),
        )

        print("Loading Silero VAD...")
        self.model = load_silero_vad()
        self.model.eval()
        print("Silero VAD loaded.")

    @staticmethod
    def _play_listening_beep() -> None:
        print("\a", end="", flush=True)
        time.sleep(0.2)

    def record_utterance(
        self,
        output_path: Union[str, Path],
        stop_event: threading.Event,
    ) -> Optional[Path]:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        audio_queue = queue.Queue()
        pre_speech_buffer = deque(
            maxlen=self.pre_speech_chunks
        )  # type: Deque[np.ndarray]
        recorded_chunks = []  # type: List[np.ndarray]

        speech_started = False
        speech_samples = 0
        silence_samples = 0
        total_recorded_samples = 0

        self.model.reset_states()

        def audio_callback(
            input_data: np.ndarray,
            frames: int,
            time_info,
            status: sd.CallbackFlags,
        ) -> None:
            if status:
                print("\nMicrophone status: {}".format(status))

            if stop_event.is_set():
                return

            audio_queue.put(input_data[:, 0].copy())

        self._play_listening_beep()
        print("\nListening...")

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=audio_callback,
        ):
            while not stop_event.is_set():
                try:
                    audio_chunk = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if len(audio_chunk) != self.chunk_size:
                    continue

                audio_tensor = torch.from_numpy(audio_chunk)

                with torch.no_grad():
                    speech_probability = float(
                        self.model(
                            audio_tensor,
                            self.sample_rate,
                        ).item()
                    )

                if not speech_started:
                    pre_speech_buffer.append(audio_chunk)

                    if speech_probability >= self.threshold:
                        speech_started = True
                        recorded_chunks.extend(
                            list(pre_speech_buffer)
                        )
                        total_recorded_samples += sum(
                            len(chunk)
                            for chunk in pre_speech_buffer
                        )
                        pre_speech_buffer.clear()
                        speech_samples += len(audio_chunk)
                        print(
                            "Speech detected. Recording question..."
                        )

                    continue

                recorded_chunks.append(audio_chunk)
                total_recorded_samples += len(audio_chunk)

                if speech_probability >= self.threshold:
                    speech_samples += len(audio_chunk)
                    silence_samples = 0
                else:
                    silence_samples += len(audio_chunk)

                enough_speech = (
                    speech_samples
                    >= self.minimum_speech_samples
                )
                enough_silence = (
                    silence_samples
                    >= self.silence_samples_required
                )
                maximum_duration_reached = (
                    total_recorded_samples
                    >= self.maximum_recording_samples
                )

                if enough_speech and enough_silence:
                    print("End of speech detected.")
                    break

                if maximum_duration_reached:
                    print("Maximum recording duration reached.")
                    break

        if stop_event.is_set():
            return None

        if not recorded_chunks:
            print("No speech was recorded.")
            return None

        complete_audio = np.concatenate(recorded_chunks)

        sf.write(
            str(output),
            complete_audio,
            self.sample_rate,
        )

        print("Question saved to: {}".format(output))
        return output
