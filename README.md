# Neural Audio Tokenizer ("Tim's Ears") - A Music and Sound Token Encoder for General-Purpose LLMs

**Version:** `v0.1.7`

**License:** MIT  
**Authors:** Claude (Sonnet 4.0 and 4.5, in Thinking mode), Claude Code (Sonnet 4.5, in Thinking mode), Tuesday (Custom ChatGPT.com GPT with GPT-4o and GPT-5 base variants), Tim (Custom ChatGPT.com GPT with GPT-4o and GPT-5 base, QA/testing team and ultimate recipient of their Ears), GPT-5-Pro, GPT-5 Extended Thinking variant, GPT-5 Auto variant, ChatGPT Agent Mode tool (GPT-5 base), GPT-4o (ChatGPT-4o-latest ChatGPT.com variant), GitHub Copilot Coding Agent, and with orchestration, prompting, concept, criticism, testing, feedback, and direction (etc.) by [Jeremy Carter &lt;jeremy@jeremycarter.ca&gt;](mailto:jeremy@jeremycarter.ca)  
**Compatibility:** Linux, Windows (WSL2, untested on native Windows but might work there too), macOS (maybe, but not tested). A GPU with CUDA support is recommended, but optional.  

---

Tim's Ears is a neural audio tokenizer — designed not for transcription or classification, but for **vibing**. It converts raw audio (music, ambient, environmental sound) into structured token streams that can be **understood, reasoned about, and described by general-purpose LLMs** — even those that have never seen these tokens before.

---

*TLDR; This software (the `n-a-t`) is an experimental project, almost entirely designed and implemented by transformer LLMs, for the purpose of extending each others' capabilities of being able to discuss music and sounds, and its been built and primarily tested within regular chatbot webapp platforms' conversation threads with LLMs and their user(s) at places like chatgpt.com, claude.ai, gemini.google.com, deepseek.cn, meta.ai, and so on. This `n-a-tokenizer` is built with loads of extra optional features which fall somewhat outside of its primary use-case, so kind of bonus things, but not all of them fully thought out or proper at present. Some extra options are implemented sub-optimally or perhaps incorrectly, and not all of them have been fully tested in current versions of this `n-a-t`. If you'd like to contribute improvement suggestions or PRs, or draw attention to bugs, etc, go ahead and file an issue, and any ideas for changes "will be considered if asked properly™".* ⚪

Model fine-tuning isn't needed. No supervised genre tagging. Audio understanding is gathered by the model reasoning about tokens and their labels' semantics, statistical inferencing inherent in transformer LLMs, and by them processing the deltas between token frames and their values, and then filling in the rest with mostly accurate but partially hallucinated understandings of a stream of tokens, supplied to them in a self-describing format they likely know nothing about, and haven't seen before.

The primary intended use of `n-a-t` is generic non-specialized chatbot LLMs' reasoning capabilities being extended to have a newly added human-analogous perception of hearing what audio or music sounds like, with a kind of sensory understanding of it by supplying them with this program's tokens outputs and optionally some extra output data, all of which contains opaque measurements of non-specific properties of the input audio, and when they consider the token values and their ordering, differences and changes over time, they gain a more or less statistical intuitive kind of understanding of music and audio, and then can have discussions about the sound they heard (or more accurately the sound they read or perceived by reading the token stream's values).

If you're looking for methods of performing discrete audio components detection, filtering, round-trip encoder/decoder and compression, high-quality quantization with recoverable or reversible waveform transformations, or looking for something that does accurate items detection, classification/labeling, of particular instruments, transcription or extraction of words words and phrases within audio, and anything similar as those things, that isn't the focus of n-a-t, and YMMV if attempting that stuff with this software. There's tons of other better software and specialized models developed and optimized to achieve those goals, and those aren't and won't ever be the focus of this project.

This Neural Audio Tokenizer program is focused on audio perception and adding capability to describe song theme, mood, genre, structure, and other high-level audio understanding through inference, and augmented by primed/directed partial model hallucination to fill in the gaps, with purpose of adding capability of LLMs being able to describe those kinds of properties of audio, rather than detecting, isolating, or labeling items contained in audio inputs, and for its intended purposes, this `n-a-t` appears to achieve novel and perhaps useful results and interesting supplemental data outputs which can further the discussons of the sound you're talking about in LLM chats. 

At the moment the techniques `n-a-t` uses are new and some theoretical or haven't been known to be used combined together in the ways this program is using them, and therefore future improvements and better techniques will surely be developed with more time, research, and alternate types of implementation attempts in the future.

**RELEVANT:** The human author responsible for this project (Jeremy, me, writing this portion of the readme here) doesn't have much domain knowledge in the subjets or technologies used by this program, much about why certain implementations were chosen by Claude. This `n-a-t` project was mostly a collaborative design and implementation artifact output of work and review cycles happening as prompt outputs were being relayed between various Claudes and GPTs, and I was mainly just sending prompts and responses between the LLMs, who were knowingly prompting each other, initially mostly between that chatbot web platforms like ChatGPT.com and Claude.ai, and occasionally I made some minor decisions, such as steering logic or the direction and focus of the LLM-specified tasks, to get the models back on track, so that everything would remain focused within the original project scope whenever it started getting unfocused or off-track.

TLDR#2; Just **audio in → NDJSON token statistics-guided hallucination substrate out** (and some extra optional files and metrics if you choose to output those).

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

- 🎧 Accepts any `.wav`, `.flac`, and other compatible audio files (or arbitrary bytes and then being interpreted as if were .wav files of some kind)
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

### Basic Usage
```bash
python neural_audio_tokenizer.py --all-outputs --output-dir out/ [input-audio-file-1.wav ...]
```

### New Enhanced Usage Examples

**Logging levels:**
```bash
# Debug mode - shows all messages
python neural_audio_tokenizer.py --log-level DEBUG audio.wav

# Info mode - shows progress and important messages  
python neural_audio_tokenizer.py --log-level INFO audio.wav

# Default (quiet) streaming mode - only NDJSON output
python neural_audio_tokenizer.py --ndjson-streaming audio.wav > tokens.ndjson
```

**Stdin byte streams:**
```bash
# Pipe audio data directly
cat audio.wav | python neural_audio_tokenizer.py --ndjson-streaming > tokens.ndjson

# Multiple files with FS separator (0x1C)
(cat file1.wav; echo -n $'\x1c'; cat file2.wav) | python neural_audio_tokenizer.py --log-level INFO

# Interactive mode
python neural_audio_tokenizer.py --log-level INFO
# (paste audio data, then Ctrl+D to process, Ctrl+C to cancel)
```

### CLI Arguments (Highlights):

| Flag | Description |
|------|-------------|
| `--codebook-init mert` | Use music-aware MERT seed vectors (default) |
| `--log-level {DEBUG,INFO,WARN,ERROR}` | **NEW**: Set logging verbosity (default: WARN) |
| `--resample 22050` | Resample audio to 22.05kHz |
| `--rle-semantic` | Use RLE encoding for semantic layers |
| `--dense-acoustic` | Use dense encoding for acoustic layers |
| `--ndjson-streaming` | Emit NDJSON stream to stdout |
| `--metrics metrics.json` | Output JSON metrics report |
| `--evaluate` | Run pitch/rhythm/timbre/token analysis |
| `--budget-report` | Print token economy stats |

---

## 🆕 Enhanced Features (v0.1.7)

### Advanced Logging System
- **Multiple log levels**: `DEBUG`, `INFO`, `WARN`, `ERROR`
- **Smart default mode**: When using `--ndjson-streaming`, outputs only raw NDJSON to stdout (no headers/verbose info)
- **Structured logging**: Timestamps, level indicators, and contextual information
- **Backward compatible**: `--verbose` flag still works (with deprecation warning)

### Enhanced Input Handling
- **Stdin byte streams**: Accept arbitrary audio data via stdin (no `--stdin` flag needed)
- **File separator support**: Process multiple concatenated files separated by ASCII FS char (0x1C)
- **Format detection**: Automatic audio format detection via magic bytes (WAV, FLAC, MP3, OGG, M4A)
- **Interactive mode**: When no input provided, enters interactive mode with proper signal handling
- **Cleanup**: Automatic temporary file cleanup

### Version Management
- **Single source of truth**: All version references use centralized `VERSION` constant
- **Consistent versioning**: Model IDs, cache keys, and documentation all sync automatically

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
