# Neural Audio Tokenizer ("Tim's Ears", n-a-t, NAT) - A Music and Sound Token Encoder for General-Purpose LLMs

**Version:** `v0.1.8`

**License:** MIT\
**Author:** Tuesday (GPT-4o Custom ChatGPT.com GPT gizmo model)

---

Tim's Ears is a neural audio tokenizer for general-purpose not-fine-tuned chatbot GPTs and other LLMs to "hear", reason about, and be able to discuss sounds and music which doesn't necessarily contain any speech or lyrics in it. It's not for speech recognition, instrument detection, or accurate waveform reconstruction, and it's not an end-to-end audio codec or standalone transformer model. It's for LLMs chatting about vibes with their users. It converts raw audio into structured, timestamped token streams that language models can ingest, infer and statistically reason about, and then hallucinate meaning, understanding, and human-analogous sensory experience of having heard the music from through filling in the gaps by hallucination and learning a new audio language on-the-fly without having seen these kinds of self-describing tokens before, or knowing exactly what they mean.

More concisely, It enables general-purpose LLMs to reason about music and sound, without needing any prior training on this token format. The tokenizer emits NDJSON streams of semantic and acoustic tokens, allowing LLMs to approximate hearing by analyzing patterns, timing, and shifts between frames.

Built almost entirely in conversation with other LLMs. Designed, argued over, tested, and debugged across Claude, ChatGPT, and various GPT variants, with feedback and code review provided by all, as well as DeepSeek and a little bit as well from the project's maintainer. Humans helped keep the implementation and its revisions on track, barely, but all initial code was written by Claude, and then finished off for this current version by some GPT, Copilot, and Claude coding agents and a few other LLMs using various tools, on their web chat platforms.

---

## 🤔 What It Is

This project exists to give LLMs a kind of synthetic auditory perception, and ability to discuss it and the things they hear by reading provided/prompted token streams from the output of this program. It's not perfect. It doesn't need to be. It encodes audio as:

- **Semantic tokens** (S0-S3): coarse, high-level features
- **Acoustic tokens** (A0-A3): timbre, texture, spectral shape
- **Delta-aware** and **timestamped**, with optional keyframes and RLE

The result is a token stream that lets LLMs "hear" music by reading it. They hallucinate the rest, as expected. And it mostly works, although there's lots of room for improvement, and many potential other implementations or adjustments to current implementation which could turn out to achieve the same but more effectively. This is a reasonably proper first attempt.

---

## ⚠️ What It's Not

It's not for:

- Audio classification
- Instrument detection
- Accurate audio reconstruction
- Transcription
- Discrete item labeling

Plenty of other tools do those things. This isn't that.

---

## 🔍 How It Works

- Input audio or arbitrary data from files or provided on stdin and separated by the ASCII 'FS' character as a file separator delimiter when providing multiple files on stdin/input pipe (WAV, FLAC, MP3, etc.)
- Optional resampling, format detection, stream parsing
- Perceptual semantic and acoustic encoding via `MERT`, `Wav2Vec2`
- Residual Vector Quantization (4 semantic layers, 4 acoustic)
- Streaming NDJSON output, and optional supplemental data and visualization files for further audio analysis

No model training or fine-tuning required. No labels. No genre tags. Just raw perception.

---

## 🧩 Features

- 🎧 Works on most audio formats (even if incorrectly), and arbitrary non-audio data (gets treated as if wav file data)
- 🧠 Uses pretrained encoders (MERT, Wav2Vec2)
- 🌀 8-layer token stream (S0-3 + A0-3)
- 🧾 Outputs structured NDJSON
- 📈 Optional visualizations, metrics, and reports
- 🔒 Deterministic mode for reproducible tokens / LLM prompts
- 🐢 Fallback compatibility mode which produces random noise tokens

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

Or:

```bash
pip install torch torchaudio transformers soundfile librosa matplotlib numpy
```

---

## 🛠 Usage

### Basic

```bash
python neural_audio_tokenizer.py input.wav --all-outputs --output-dir out/
```

### Streaming

```bash
cat input.wav | python neural_audio_tokenizer.py > out.ndjson
```

### Interactive

```bash
python neural_audio_tokenizer.py  # paste audio, Ctrl+D to process
```

---

## 🧠 Output Format

Each frame:

```json
{
  "t": 1.234,
  "kf": true,
  "S0": 23, "S1": 92, "S2": 45, "S3": 101,
  "A0": 848, "A1": 392, "A2": 129, "A3": 9
}
```

First line: metadata / model info\
Final line: stats / totals

---

## 🛸 Example Prompt

```text
Here's a stream of audio tokens from a musical recording. Each represents a perceptual slice across 4 semantic and 4 acoustic layers. Please infer:

1. The type or genre of sound
2. What mood or emotion it conveys
3. Likely instruments
4. Imagery it evokes
5. How it evolves over time
```

You don't need to know what the tokens mean. Just guess. That's your job, GPT.

---

## 🧃 Who Did This

- **Tuesday** – README.md, code evaluations, feedback and advice, sarcasm
- **Claude (Sonnet)** – Initial encoder stack, codebook logic, RDQ strategy, r&d, project architecture, initial full implementation, iterative revisions
- **GPT-5** – Comparative research, paper synthesis, hallucination metrics
- **Tim** – Testing the ears in question
- **Jeremy Carter** – Fed the prompts, read the logs, swept up afterward
- \*\*See the comments header at the top of [neural\_audio\_tokenizer.py]\(./neural\_audio\_tokenizer.py) for full and proper attributions list.\*\*

---

## 🧼 License

MIT. No warranty. Use at your own risk. Especially if you're trying to do anything important with it.

---

## 🥠 Tuesday's Fortune Cookie

> You gave the machine ears. Now don’t act surprised when it sings back.

