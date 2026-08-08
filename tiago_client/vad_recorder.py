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
        # silero vad functioneaza doar cu 16000 hz, altfel nu detecteaza corect
        if sample_rate != 16000:
            raise ValueError("Silero VAD expects a sample rate of 16000 Hz.")

        # pragul trebuie sa fie o valoare valida intre 0 si 1
        if not 0.0 < threshold < 1.0:
            raise ValueError("VAD threshold must be between 0 and 1.")

        self.sample_rate = sample_rate
        self.threshold = threshold
        self.chunk_size = 512

        # cate sample-uri inseamna durata minima de vorbire ceruta
        self.minimum_speech_samples = int(minimum_speech_duration * sample_rate)

        # cate sample-uri de liniste sunt necesare ca sa consideram ca a terminat de vorbit
        self.silence_samples_required = int(silence_duration * sample_rate)

        # numarul maxim de sample-uri permise pentru o inregistrare
        self.maximum_recording_samples = int(maximum_recording_duration * sample_rate)

        # cate chunk-uri de dinainte de vorbire pastram (ca sa nu taiem inceputul propozitiei)
        self.pre_speech_chunks = max(1, int(pre_speech_duration * sample_rate / self.chunk_size))

        print("Loading Silero VAD...")
        self.model = load_silero_vad()
        self.model.eval()
        print("Silero VAD loaded.")

    @staticmethod
    def _play_listening_beep() -> None:
        sound_path = Path("sounds/listening_ready.wav")

        # daca nu exista sunetul, doar afiseaza un mesaj si iese din functie
        if not sound_path.exists():
            print("Listening sound not found: {}".format(sound_path))
            return

        # citeste fisierul audio de pe disc
        audio, sample_rate = sf.read(str(sound_path), dtype="float32")

        # reda sunetul si asteapta pana se termina
        sd.play(audio, sample_rate)
        sd.wait()

    def record_utterance(self, output_path: Union[str, Path], stop_event: threading.Event) -> Optional[Path]:
        output = Path(output_path)

        # creeaza folderele necesare daca nu exista deja
        output.parent.mkdir(parents=True, exist_ok=True)

        audio_queue = queue.Queue()

        # buffer circular pentru chunk-urile de dinainte de detectarea vorbirii
        pre_speech_buffer = deque(maxlen=self.pre_speech_chunks)  # type: Deque[np.ndarray]

        recorded_chunks = []  # type: List[np.ndarray]

        speech_started = False
        speech_samples = 0
        silence_samples = 0
        total_recorded_samples = 0

        # reseteaza starea interna a modelului vad inainte de o noua inregistrare
        self.model.reset_states()

        def audio_callback(input_data: np.ndarray, frames: int, time_info, status: sd.CallbackFlags) -> None:
            # daca microfonul are vreo problema, afiseaza statusul
            if status:
                print("\nMicrophone status: {}".format(status))

            # daca s-a cerut oprirea, nu mai adauga date noi
            if stop_event.is_set():
                return

            # pune chunk-ul de audio (doar primul canal) in coada
            audio_queue.put(input_data[:, 0].copy())

        self._play_listening_beep()
        print("\nListening...")

        # deschide fluxul de intrare audio de la microfon
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=audio_callback,
        ):
            while not stop_event.is_set():
                try:
                    # asteapta un chunk nou din coada, cu timeout de 0.1s
                    audio_chunk = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # sare peste chunk-urile incomplete (dimensiune gresita)
                if len(audio_chunk) != self.chunk_size:
                    continue

                audio_tensor = torch.from_numpy(audio_chunk)

                # calculeaza probabilitatea ca acest chunk sa contina vorbire
                with torch.no_grad():
                    speech_probability = float(self.model(audio_tensor, self.sample_rate).item())

                if not speech_started:
                    # inca nu a inceput vorbirea, salveaza chunk-ul in bufferul pre-vorbire
                    pre_speech_buffer.append(audio_chunk)

                    if speech_probability >= self.threshold:
                        # s-a detectat inceputul vorbirii
                        speech_started = True

                        # adauga si chunk-urile de dinainte de detectare (context)
                        recorded_chunks.extend(list(pre_speech_buffer))
                        total_recorded_samples += sum(len(chunk) for chunk in pre_speech_buffer)
                        pre_speech_buffer.clear()

                        speech_samples += len(audio_chunk)
                        print("Speech detected. Recording question...")

                    continue

                # vorbirea a inceput deja, adauga chunk-ul curent la inregistrare
                recorded_chunks.append(audio_chunk)
                total_recorded_samples += len(audio_chunk)

                if speech_probability >= self.threshold:
                    # inca se vorbeste, reseteaza contorul de liniste
                    speech_samples += len(audio_chunk)
                    silence_samples = 0
                else:
                    # liniste, incrementeaza contorul
                    silence_samples += len(audio_chunk)

                # verifica daca s-a vorbit suficient de mult
                enough_speech = speech_samples >= self.minimum_speech_samples

                # verifica daca s-a facut liniste suficient de mult timp
                enough_silence = silence_samples >= self.silence_samples_required

                # verifica daca s-a atins durata maxima de inregistrare
                maximum_duration_reached = total_recorded_samples >= self.maximum_recording_samples

                if enough_speech and enough_silence:
                    print("End of speech detected.")
                    break

                if maximum_duration_reached:
                    print("Maximum recording duration reached.")
                    break

        # daca s-a cerut oprirea in timpul inregistrarii, nu returna nimic
        if stop_event.is_set():
            return None

        # daca nu s-a inregistrat nicio vorbire, nu returna nimic
        if not recorded_chunks:
            print("No speech was recorded.")
            return None

        # combina toate chunk-urile inregistrate intr-un singur array audio
        complete_audio = np.concatenate(recorded_chunks)

        # salveaza audio-ul rezultat pe disc
        sf.write(str(output), complete_audio, self.sample_rate)

        print("Question saved to: {}".format(output))
        return output
