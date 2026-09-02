# TIAGo Voice Assistant

Audio communication with the PAL Robotics TIAGo robot using Automatic Speech Recognition, contextual knowledge retrieval, a Large Language Model and Romanian Text-to-Speech.

---

## Description

This project implements a voice-based conversational assistant for the PAL Robotics TIAGo robot.

The system allows the user to communicate with TIAGo using natural speech. The robot records the user's voice, detects the speech segment using Voice Activity Detection, sends the recorded audio to an external processing server, converts the speech into text, retrieves relevant information from a local knowledge base, generates a response using Gemini, converts the response into Romanian speech using Piper TTS and sends the generated audio back to the robot.

The complete pipeline is:

```text
User
  |
  v
TIAGo Microphone
  |
  v
Silero VAD
  |
  v
Audio Recording
  |
  v
SpeD Parakeet ASR
  |
  v
Context Selection
  |
  v
Knowledge Retrieval
  |
  v
Prompt Builder
  |
  v
Gemini LLM
  |
  v
Sentence Streaming
  |
  v
Piper TTS
  |
  v
Audio Chunks
  |
  v
TIAGo Speaker
```

---

## Features

- Voice interaction with TIAGo
- Voice Activity Detection using Silero VAD
- Automatic Speech Recognition using SpeD Parakeet
- Romanian speech recognition
- Conversation history
- Context selection
- Local knowledge retrieval using a Mini-RAG architecture
- Gemini-based response generation
- Sentence-level response streaming
- Romanian Text-to-Speech using Piper
- Audio chunk generation and transmission
- Direct TCP/IP communication with the processing server
- TCP/IP communication with the processing server
- Separation between robot-side and computationally intensive processing

---

## System Architecture

The system is divided into two main parts:

1. TIAGo Client
2. Processing Server

### TIAGo Client

The TIAGo client runs directly on the robot.

Its responsibilities are:

- microphone input
- speech detection
- audio recording
- TCP/IP communication
- receiving generated audio
- audio playback
- communication status handling

### Processing Server

The processing server runs on an external computer.

Its responsibilities are:

- receiving recorded audio
- speech recognition
- context selection
- knowledge retrieval
- conversation management
- prompt construction
- Gemini response generation
- sentence-level streaming
- Romanian speech synthesis
- sending generated audio back to TIAGo

---

## Communication Pipeline

The complete communication process is:

1. The user starts a conversation with TIAGo.
2. The TIAGo microphone captures the user's speech.
3. Silero VAD detects when the user starts and stops speaking.
4. The recorded speech is saved as a WAV file.
5. The audio is sent to the processing server.
6. SpeD Parakeet converts the speech into text.
7. The system determines the relevant conversational context.
8. The Retriever searches the local knowledge base.
9. The retrieved information is combined with the user's question and conversation history.
10. Gemini generates the response.
11. The response is processed sentence by sentence.
12. Each sentence is converted into Romanian speech using Piper TTS.
13. The generated audio chunks are sent back to TIAGo.
14. TIAGo plays the response through its speaker.

---

## Voice Activity Detection

The TIAGo client uses Silero VAD to detect speech automatically.

Current configuration:

| Parameter | Value |
|---|---:|
| Sample rate | 16000 Hz |
| Channels | Mono |
| Speech threshold | 0.5 |
| Minimum speech duration | 0.4 s |
| Silence duration | 1.2 s |
| Maximum recording duration | 30 s |
| Pre-speech duration | 0.3 s |

VAD allows the system to determine when the user is speaking instead of continuously recording audio.

---

## Automatic Speech Recognition

The processing server uses SpeD Parakeet for Automatic Speech Recognition.

Default configuration:

| Parameter | Value |
|---|---|
| Model | `base` |
| Input language | Romanian |

The processing flow is:

```text
Audio
  |
  v
SpeD Parakeet
  |
  v
Romanian transcription
  |
  v
Context and knowledge processing
```

---

## Context Selection

The system includes a Context Selector that determines the relevant laboratory context for the current conversation.

This allows the same conversational pipeline to work with different laboratory-specific information.

```text
User Question
      |
      v
Context Selector
      |
      v
Relevant Laboratory Context
```

---

## Knowledge Retrieval

The project uses a local knowledge base stored in:

```text
knowledge/
```

The Retriever searches this knowledge base and selects the most relevant information for the current question.

The current configuration uses:

```text
Top-K:
    3

Minimum retrieval score:
    0.18
```

The retrieved information is passed to the Prompt Builder together with the question and conversation history.

This creates a simple Retrieval-Augmented Generation pipeline that allows Gemini to answer questions using laboratory-specific information.

---

## Conversation Management

The assistant maintains conversation history in order to support multi-turn interactions.

The current configuration stores up to:

```text
10 messages
```

This allows follow-up questions to be interpreted using previous turns.

Example:

```text
User:
What is a diode?

TIAGo:
A diode is an electronic component...

User:
What is its main application?

TIAGo:
Its main application is...
```

---

## Gemini LLM

Gemini is used as the Large Language Model responsible for generating the assistant's responses.

The prompt is constructed using:

```text
User Question
+
Conversation History
+
Selected Context
+
Retrieved Knowledge
+
Response Instructions
```

The general flow is:

```text
User Question
      +
Conversation History
      +
Selected Context
      +
Retrieved Knowledge
      |
      v
Prompt Builder
      |
      v
Gemini
      |
      v
Generated Response
```

---

## Sentence-Level Streaming

Instead of waiting for Gemini to finish the entire response before starting speech synthesis, the system processes the response incrementally.

```text
Gemini
  |
  +---- Sentence 1 ---> Piper ---> Audio Chunk 1 ---> TIAGo
  |
  +---- Sentence 2 ---> Piper ---> Audio Chunk 2 ---> TIAGo
  |
  +---- Sentence 3 ---> Piper ---> Audio Chunk 3 ---> TIAGo
```

This reduces perceived response latency because TIAGo can start speaking while the rest of the response is still being generated.

---

## Text-to-Speech

Romanian speech synthesis is performed using Piper TTS.

Default model:

```text
models/piper/ro_RO-lili-high.onnx
```

Each generated sentence is synthesized independently.

Generated audio chunks are stored in:

```text
samples/output/answer_chunks/
```

---

## Network Communication

The TIAGo client communicates directly with the processing server using TCP/IP.

The communication flow is:

```text
TIAGo
  |
  | TCP/IP
  |
  v
Processing Server
```

The robot sends the recorded question audio to the server and receives the generated response audio through the same network connection.

---

## Client Status

The TIAGo client uses the following status states:

```text
LISTENING
RECORDING_ERROR
UPLOADING
PROCESSING
COMMUNICATION_ERROR
PLAYING
PLAYBACK_ERROR
```

The normal status flow is approximately:

```text
LISTENING
    |
    v
Recording
    |
    v
UPLOADING
    |
    v
PROCESSING
    |
    v
PLAYING
    |
    v
LISTENING
```

---

## Network Communication

The processing server uses TCP/IP communication.

Default configuration:

```text
Host:
    0.0.0.0

Port:
    5000
```

The server listens for an incoming TIAGo client connection and processes the received audio.

The robot-side robot-side components and the external processing server are separated so that computationally intensive ASR, retrieval, LLM and TTS operations can run on a more powerful computer.

---

## Audio Flow

### Question Audio

```text
TIAGo Microphone
      |
      v
Silero VAD
      |
      v
question.wav
      |
      v
TCP/IP
      |
      v
Processing Server
      |
      v
SpeD Parakeet
```

### Response Audio

```text
Gemini
      |
      v
Generated Sentence
      |
      v
Piper TTS
      |
      v
Audio Chunk
      |
      v
Network
      |
      v
TIAGo
      |
      v
Speaker
```

---

## Project Structure

```text
Audio-communication-with-TIAGo-using-ASR-TTS-and-Voice-Cloning/
|
+-- knowledge/
|   +-- Laboratory knowledge files
|
+-- samples/
|   +-- input/
|   |   +-- received_question.wav
|   |
|   +-- output/
|       +-- answer_chunks/
|
+-- tests/
|
+-- tiago_assistant/
|   +-- asr.py
|   +-- context_selector.py
|   +-- conversation.py
|   +-- dialog.py
|   +-- network.py
|   +-- prompt_builder.py
|   +-- retriever.py
|   +-- tts.py
|   +-- ...
|
+-- tiago_client/
|   +-- TIAGo TCP/IP client
|
+-- main.py
+-- main_server.py
+-- main_text.py
+-- test_tts.py
|
+-- .env.example
+-- environment.yml
+-- requirements.txt
+-- requirements-server.txt
+-- README.md
```

---

## Configuration

The processing server provides the following main command-line options:

| Option | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | Server listening address |
| `--port` | `5000` | TCP server port |
| `--model` | `base` | SpeD Parakeet ASR model |
| `--input-language` | `ro` | Input speech language |
| `--response-language` | `ro` | Response language |
| `--knowledge` | `knowledge` | Knowledge base directory |
| `--top-k` | `3` | Number of retrieved results |
| `--received-audio` | `samples/input/received_question.wav` | Received audio path |
| `--tts-model` | `models/piper/ro_RO-lili-high.onnx` | Piper TTS model |
| `--answer-chunks-directory` | `samples/output/answer_chunks` | Generated audio chunks directory |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Davyyd21/Audio-communication-with-TIAGo-using-ASR-TTS-and-Voice-Cloning.git

cd Audio-communication-with-TIAGo-using-ASR-TTS-and-Voice-Cloning
```

Create the Conda environment:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate <environment-name>
```

Alternatively, install the Python dependencies using:

```bash
pip install -r requirements.txt
```

For the processing server:

```bash
pip install -r requirements-server.txt
```

---

## Environment Variables

The repository contains an example environment configuration:

```text
.env.example
```

Create the local configuration:

```bash
cp .env.example .env
```

Configure the required API keys and environment-specific settings inside `.env`.

The `.env` file should not be committed to the repository.

---

## Running the Processing Server

Start the processing server with:

```bash
python main_server.py
```

The default configuration is:

```text
Host:
    0.0.0.0

Port:
    5000

ASR model:
    base

Input language:
    Romanian

Response language:
    Romanian

Knowledge directory:
    knowledge

Top-K:
    3

TTS model:
    models/piper/ro_RO-lili-high.onnx
```

A custom configuration can be provided using command-line arguments:

```bash
python main_server.py     --host 0.0.0.0     --port 5000     --model base     --input-language ro     --response-language ro     --knowledge knowledge     --top-k 3
```

---

## Running the TIAGo Client

The TIAGo-side application must be executed directly on the robot in a robot environment.

The client:

2. Initializes the microphone and VAD.
3. Waits for user interaction.
4. Records the user's speech.
5. Sends the question.
6. Waits for the generated response.
7. Receives the audio response.
8. Plays the response through TIAGo's speaker.

---

## Knowledge Base

Laboratory-specific information is stored in:

```text
knowledge/
```

The knowledge base can be modified independently from the main conversational pipeline.

This allows new laboratory information to be added without changing the ASR, LLM or TTS components.

---

## Design Goals

The project focuses on:

- low perceived response latency
- Romanian speech interaction
- contextual answers
- laboratory-specific knowledge
- modular architecture
- separation between robot-side and processing-side workloads
- real-time conversational interaction
- integration with the TIAGo robot environment

The computationally intensive tasks are handled by the external processing computer, while TIAGo is primarily responsible for audio acquisition, TCP/IP communication and playback.

---

## Technologies

| Component | Technology |
|---|---|
| Robot | PAL Robotics TIAGo |
| Communication | TCP/IP |
| Voice Activity Detection | Silero VAD |
| Speech Recognition | SpeD Parakeet |
| Knowledge Retrieval | Mini-RAG |
| Language Model | Gemini |
| Text-to-Speech | Piper |
| Communication | TCP/IP |
| Audio Format | WAV |
| Programming Language | Python |

---

## Repository

GitHub repository:

https://github.com/Davyyd21/Audio-communication-with-TIAGo-using-ASR-TTS-and-Voice-Cloning
