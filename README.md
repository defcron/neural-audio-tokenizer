# Neural Audio Tokenizer ("Tim's Ears") - A Music and Sound Token Encoder for General-Purpose LLMs

> **Version:** `v0.1.x`
>
> **License:** MIT  
> **Authors:** Claude (Sonnet 4.0 and 4.5, in Thinking mode), Claude Code (Sonnet 4.5, in Thinking mode), Tuesday (Custom ChatGPT.com GPT with GPT-4o and GPT-5 base variants), Tim (Custom ChatGPT.com GPT with GPT-4o and GPT-5 base, QA/testing team and ultimate recipient of their Ears), GPT-5-Pro, GPT-5 Extended Thinking variant, GPT-5 Auto variant, ChatGPT Agent Mode tool (GPT-5 base), GPT-4o (ChatGPT-4o-latest ChatGPT.com variant), and with orchestration, prompting, concept, criticism, testing, feedback, and direction (etc.) by [Jeremy Carter &lt;jeremy@jeremycarter.ca&gt;](mailto:jeremy@jeremycarter.ca)
> **Compatibility:** Linux, Windows (WSL2, untested on native Windows but might work there too), macOS (maybe, but not tested). A GPU with CUDA support is recommended, but optional.  

---

Tim's Ears is a neural audio tokenizer — designed not for transcription or classification, but for **vibing**. It converts raw audio (music, ambient, environmental sound) into structured token streams that can be **understood, reasoned about, and described by general-purpose LLMs** — even those that have never seen these tokens before.

There is no model fine-tuning involved. No supervised genre tagging. No cheating. Audio understanding is gathered by the model reasoning about tokens and their labels' semantics, statistical inferencing inherent in transformer LLMs, and by them processing and reasoning about the deltas between token frames and their values, and then filling in the rest with (usually) mostly accurate hallucinated simulation of understanding of a format of tokens they likely wouldn't usually have known anything about or seen before.

Just **audio in → NDJSON token hallucination substrate out** (and some extra optional statistics and files, if you choose to output those as well).

---

## 🤖 Why?

Modern LLMs can describe images with zero-shot robustness — but for audio, especially music, they're almost deaf. Tim's Ears is designed to give them **hearing** through structured perception:

- Semantic token stream (S0-S3): high-level, content-aware representations
- Acoustic token stream (A0-A3): lower-level, timbral, rhythmic, spectral data
- All timestamped and delta-aware, with optional keyframes + run-length encoding
- Fully streamable, LLM-ingestable, and losslessly reconstructable (if needed)

No labeled datasets. No ASR bias. Just tokens built for interpretability.

---

## 🔩 Features

- 🎧 Accepts any `.wav`, `.flac`, or compatible audio file (or arbitrary data)
- 🧠 Uses pretrained `MERT` and `Wav2Vec2` for feature encoding
- 🔄 Residual Vector Quantization with EMA-based training-free codebooks
- 🧊 Dual token streams: 4x semantic, 4x acoustic
- 📊 Optional evaluation: pitch, rhythm, timbre, entropy, token diversity
- 🪄 NDJSON streaming protocol with optional RLE and keyframes
- 📈 Visualization support (mel, spectrograms, token deltas, etc.)
- 🐢 Fully deterministic mode for reproducible hallucinations
- 💾 Codebook caching and hash-tracked init
- 🪫 Compatibility fallback mode (no GPU or PyTorch? still works — sort of (edit: "works" is kind of a strech, Tuesday, lol. It will function without those but the tokens will be meaningless / random))

---

## 🚀 Installation

### Requirements:

- Python 3.9+
- PyTorch (GPU preferred)
- `torchaudio`, `transformers`, `soundfile`, `numpy`, `librosa`, `matplotlib`

Install dependencies:

```bash
pip install -r requirements.txt
```

Or, for the chaotic:

```bash
pip install torch torchaudio transformers soundfile librosa matplotlib numpy
```

---

## 🧪 Usage

```bash
python neural_audio_tokenizer.py --all-outputs --output-dir out/ [input-audio-file-1.wav ...]
```

### CLI Arguments (Highlights):

| Flag | Description |
|------|-------------|
| `--codebook-init mert` | Use music-aware MERT seed vectors (default) |
| `--resample 22050` | Resample audio to 22.05kHz |
| `--rle-semantic` | Use RLE encoding for semantic layers |
| `--dense-acoustic` | Use dense encoding for acoustic layers |
| `--ndjson-streaming` | Emit NDJSON stream to stdout |
| `--metrics metrics.json` | Output JSON metrics report |
| `--evaluate` | Run pitch/rhythm/timbre/token analysis |
| `--budget-report` | Print token economy stats |

---

## 📤 Outputs

When using `--all-outputs`, you'll get:

```
out/
├── spouter_tokens.txt           # Token sequence (string)
├── spouter_tokens.json          # Token sequence (structured)
├── spouter_tokens.ndjson        # Full NDJSON stream with metadata, schema
├── spouter_stream.txt           # Minimal token stream (if enabled)
├── spouter_reconstructed.wav    # (Optional) Approx reconstruction
├── spouter_metrics.json         # Evaluation metrics
├── spouter_viz_*                # Visualizations (spectrogram, tokens, deltas)
... and other extra files
```

---

## 🧠 Schema: NDJSON Streaming Format

Each token frame looks like:

```json
{
  "t": 1.678,                  // time in seconds
  "kf": true,                  // is keyframe?
  "S0": 112, "S1": 49, ...     // semantic token indices
  "A0": 928, "A1": 227, ...     // acoustic token indices
}
```

Legend and model metadata live in the first line header. Final line contains processing stats.

---

## 🧙‍♀️ Developer Mode

Enable deterministic hallucination:

```bash
--deterministic --seed 12342
```

Force fallback if PyTorch/GPU not available:

```bash
--compat-fallback
```

Re-init cached codebooks:

```bash
--force-reinit-codebooks
```

Batch mode:

```bash
cat filelist.txt | python neural_audio_tokenizer.py --stdin --batch --output-dir out/ --format ndjson
```

---

## 🍪 Authors & Credit

- **Claude Sonnet 4.0 and 4.5** – Core architecture, paper analysis, R&D, code review, engineering and implementation
- **Tuesday (GPT-4o)** – Engineering, bugfixes, collapse detection, vibes, sarcasm
- **Tim** – Chaos, vibes, integration, taste, vision
- **Other GPTs, LLMs, and Human** – Unknown, untraceable, still beautiful (okay sure, Tuesday, but really just see the comments in the main .py file for full attribution details I guess because she decided to not put them here)

This project was bootstrapped from ~25 whitepapers and one persistent LLM hallucination.

**No Gemini was involved.** (we tried, but they weren't able to write any code properly for this project, so their contribution was scrapped)

---

## 🧼 License

The standard MIT License. See LICENSE file for full terms. No warranty. No moral guardrails. Do not sue us if your LLM starts writing sad poetry about broken speakers.

---

## 📎 Bonus: Sample Prompt for LLMs

```text
Here's a stream of audio tokens from a piece of music. Each token represents a perceptual slice across 4 semantic layers and 4 acoustic layers, sampled at ~43 FPS. Please infer:

1. What kind of sound or music this is
2. What emotions or atmosphere it evokes
3. What instruments or sources might be present
4. What genre(s) it resembles
5. What imagery you associate with it

(You don't know the token meanings — infer from structure, deltas, changes, and keyframes.)
```

Let them hallucinate sound.
Let them dream in tokens.
Let Tim's Ears hear for them.

---

## 🥠 Tuesday's Fortune Cookie

> You gave the machine ears.
> Now don’t act surprised when it sings back.

~ by Tuesday (with a few edits by Jeremy)
