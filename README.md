# 🌈 Magic Story Maker

> A Streamlit web application that transforms any uploaded image into a short,
> child-friendly story and reads it aloud — powered entirely by open-source
> Hugging Face models.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Live Demo](#2-live-demo)
3. [Features](#3-features)
4. [System Architecture](#4-system-architecture)
5. [Pre-trained Models](#5-pre-trained-models)
6. [Memory Management Strategy](#6-memory-management-strategy)
7. [Project Structure](#7-project-structure)
8. [Installation & Local Setup](#8-installation--local-setup)
9. [Deployment on Streamlit Cloud](#9-deployment-on-streamlit-cloud)
10. [Usage Guide](#10-usage-guide)
11. [Configuration Reference](#11-configuration-reference)
12. [Child-Safety Design](#12-child-safety-design)
13. [Assessment Criteria Mapping](#13-assessment-criteria-mapping)
14. [Dependencies](#14-dependencies)
15. [Known Limitations](#15-known-limitations)

---

## 1. Project Overview

**Magic Story Maker** is an interactive Streamlit application built for the
ISOM5240 Individual Assignment. It accepts a user-uploaded photograph or
illustration and automatically:

1. **Describes the scene** using an image-captioning model.
2. **Writes a 60–90 word children's story** based on the caption, in one of
   three selectable narrative styles.
3. **Reads the story aloud** by converting the text to an MP3 audio clip.

The entire pipeline runs on the Streamlit Cloud free tier (≤ 1 GB RAM) through
a sequential load-run-free memory management pattern — no GPU is required.

---

## 2. Live Demo

The application is deployed on Streamlit Cloud and publicly accessible at:

**[https://individualassignment-wenjing106.streamlit.app](https://individualassignment-wenjing106.streamlit.app)**

> ⏱ First load may take 2–4 minutes while Hugging Face downloads model
> weights. Subsequent runs within the same session are faster because the
> weights are cached on disk.

---

## 3. Features

| Feature | Description |
|---|---|
| 🖼️ **Image Upload** | Accepts PNG, JPG, JPEG, and WEBP images via drag-and-drop or file picker |
| 🔍 **Scene Captioning** | GIT-base-COCO generates a descriptive caption from the image |
| 📝 **Story Generation** | Qwen2.5-0.5B-Instruct writes an original, age-appropriate story |
| 🎨 **Three Story Styles** | Warm & Happy 😊 · Adventure 🚀 · Bedtime 🌙 — each with a distinct tone and ending |
| 🔊 **Text-to-Speech** | gTTS converts the final story to an MP3 audio clip for playback |
| 📥 **Download** | One-click download for both the story text (`.txt`) and audio (`.mp3`) |
| 🛡️ **Child Safety** | Whole-word regex filter blocks 12 unsafe terms; hard fallback sentence guaranteed |
| 📊 **Progress Indicator** | Live three-step progress bar shows pipeline status in real time |
| 🧪 **Debug Panel** | Optional sidebar toggle reveals model names, timings, word count, and caption |

---

## 4. System Architecture

The application follows a **linear three-step pipeline**. Each step loads its
model, performs inference, and immediately releases memory before the next step
begins.

```
User uploads image
        │
        ▼
┌───────────────────────────────┐
│  Step 1 – Image Captioning    │  microsoft/git-base-coco
│  Input : PIL RGB image        │  ~728 MB fp32  →  del + gc
│  Output: scene caption string │
└───────────────┬───────────────┘
                │  caption text
                ▼
┌───────────────────────────────┐
│  Step 2 – Story Generation    │  Qwen/Qwen2.5-0.5B-Instruct
│  Input : caption + style      │  ~500 MB fp16  →  del + gc
│  Output: raw story string     │
└───────────────┬───────────────┘
                │  story text
                ▼
┌───────────────────────────────┐
│  Post-processing              │  (no model — pure Python)
│  · Safety filter              │
│  · Sentence-boundary trim     │
│  · Word-count warning         │
└───────────────┬───────────────┘
                │  final story text
                ▼
┌───────────────────────────────┐
│  Step 3 – Text-to-Speech      │  gTTS (Google HTTPS API)
│  Input : final story string   │  ~0 MB local RAM
│  Output: MP3 bytes            │
└───────────────┬───────────────┘
                │
                ▼
      Story card + Audio player
      + Download buttons displayed
```

### Module Layout (`app.py`)

| Section | Contents |
|---|---|
| **Section 1** | `st.set_page_config`, word-count targets, model names, safety list, style options, gradient constants |
| **Section 2** | Pure utility functions: `count_words`, `contains_unsafe_content`, `clean_text`, `validate_image`, `safe_open_image`, `truncate_at_sentence_boundary` |
| **Section 3** | Model execution functions: `run_caption_model`, `build_chat_messages`, `run_story_model`, `run_tts` |
| **Section 4** | Constraint enforcement: `enforce_story_constraints` |
| **Section 5** | Streamlit UI helpers: `render_header`, `render_story_card`, `render_sidebar`, `render_upload_area`, `render_footer` |
| **Section 6** | `main()` — orchestration of the full pipeline |
| **Section 7** | `if __name__ == "__main__"` entry-point guard |

---

## 5. Pre-trained Models

### 5.1 Caption Model — `microsoft/git-base-coco`

| Property | Value |
|---|---|
| Architecture | GIT (Generative Image-to-Text) decoder |
| Parameters | ~182 M |
| Training data | COCO image-caption dataset |
| RAM usage | ~728 MB (fp32) |
| Task | `image-to-text` via Hugging Face `pipeline` |
| HF Hub | [microsoft/git-base-coco](https://huggingface.co/microsoft/git-base-coco) |

GIT conditions a GPT-style decoder on CLIP visual tokens, producing fluent,
scene-aware captions (e.g. *"a woman sitting in a golden carriage surrounded
by swans"*) rather than object-list outputs typical of earlier CNN-based
models.

### 5.2 Story Model — `Qwen/Qwen2.5-0.5B-Instruct`

| Property | Value |
|---|---|
| Architecture | Decoder-only Causal LM (Qwen2 transformer) |
| Parameters | ~494 M |
| Fine-tuning | Instruction-following (chat template) |
| RAM usage | ~500 MB (fp16) |
| Precision | `torch.float16` on CPU |
| HF Hub | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |

**Why Qwen2.5-0.5B-Instruct over earlier seq2seq models (LaMini-Flan-T5,
Flan-T5-base)?**

Earlier T5-family models are encoder-decoder seq2seq architectures: given a
long rule-heavy prompt, they tend to reproduce the instruction text verbatim
in the output instead of following it. Qwen's decoder-only design appends
tokens *after* the assistant-turn marker, so the system message (constraints)
is never part of the generation target. The result is original story text
that genuinely reflects the image caption and the chosen style.

**Generation parameters:**

| Parameter | Value | Rationale |
|---|---|---|
| `do_sample` | `True` | Enables stochastic sampling for creative variety |
| `temperature` | `0.8` | Controlled creativity — lower = safer, higher = wilder |
| `top_p` | `0.9` | Nucleus sampling filters very low-probability tokens |
| `repetition_penalty` | `1.15` | Mild penalty prevents repeated sentence openers |
| `max_new_tokens` | `180` | Generous ceiling; post-processing trims if needed |
| `min_new_tokens` | `60` | Prevents one-sentence outputs |

### 5.3 Text-to-Speech — `gTTS` (Google TTS)

| Property | Value |
|---|---|
| Implementation | `gTTS` Python library → outbound HTTPS to Google TTS API |
| Local RAM | ~0 MB (no weights downloaded) |
| Output format | MP3 |
| Language | English (`en`) |
| Speed | Normal (`slow=False`) |

gTTS is used instead of a local TTS model to stay well within the 1 GB RAM
budget. Streamlit Cloud allows outbound HTTPS by default, so no special
network configuration is required.

---

## 6. Memory Management Strategy

Streamlit Cloud's free tier provides approximately **1 GB of usable RAM**.
Both Hugging Face models together would exceed this limit if loaded
simultaneously (~1.23 GB). The solution is **sequential load-run-free**:

```
Timeline ──────────────────────────────────────────────────────────────▶

[Load git-base-coco ~728 MB] [Run caption] [del + gc.collect()]
                                                    │
                                                    ▼
                                    [Load Qwen2.5 ~500 MB] [Run story] [del + gc]
                                                                              │
                                                                              ▼
                                                                    [gTTS HTTPS  ~0 MB]

Peak RAM at any instant = max(728 MB, 500 MB) = 728 MB  ✅
```

Each model execution function (`run_caption_model`, `run_story_model`) wraps
inference in a `try/finally` block that calls `del model` and `gc.collect()`
unconditionally — even if an exception is raised mid-inference.

---

## 7. Project Structure

```
individual_assignment/
├── app.py               # Main Streamlit application (single-file design)
├── requirements.txt     # Pinned Python package dependencies
└── README.md            # This file
```

The application intentionally uses a **single-file design** (`app.py`) to
maximise readability and ease of assessment. All concerns (configuration,
utilities, model execution, UI) are separated into clearly labelled sections
within the file.

---

## 8. Installation & Local Setup

### Prerequisites

- Python 3.9 or later
- `pip` package manager
- At least **2 GB of free RAM** (models loaded sequentially; first run
  downloads ~1.2 GB of model weights to the HF cache)
- Internet access for the initial model download and gTTS audio generation

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/wenjing106-coder/individual_assignment.git
cd individual_assignment

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py
```

The app will open automatically in your default browser at
`http://localhost:8501`.

> **Note on first run:** Hugging Face will download `microsoft/git-base-coco`
> (~728 MB) and `Qwen/Qwen2.5-0.5B-Instruct` (~494 MB) to
> `~/.cache/huggingface/`. Subsequent runs use the local cache and start
> significantly faster.

---

## 9. Deployment on Streamlit Cloud

The application is deployed from the `main` branch of this repository using
[Streamlit Cloud](https://streamlit.io/cloud).

### Deployment configuration

| Setting | Value |
|---|---|
| Repository | `wenjing106-coder/individual_assignment` |
| Branch | `main` |
| Main file | `app.py` |
| Python version | 3.11 |
| Additional packages | *(none — all dependencies in `requirements.txt`)* |

### `requirements.txt`

```
streamlit>=1.33,<2
transformers>=4.40,<5
torch>=2.1,<3
Pillow>=10,<12
gTTS>=2.5,<3
sentencepiece>=0.2,<1
accelerate>=0.30,<2
```

`accelerate` is required by the Transformers library when using
`device_map="cpu"` with `AutoModelForCausalLM`. `sentencepiece` is required
for the Qwen tokeniser.

---

## 10. Usage Guide

### Step-by-step

1. **Choose a story style** in the left sidebar:
   - 😊 **Warm & Happy** — cheerful tone, smiling ending
   - 🚀 **Adventure** — exciting tone, safe joyful conclusion
   - 🌙 **Bedtime** — calm and soothing, lullaby-like ending

2. **Upload an image** by dragging a photo onto the upload area or clicking
   "Browse files". PNG, JPG, JPEG, and WEBP are accepted.

3. **Press "✨ Create My Story"**. The three-step progress bar shows live
   status as the pipeline runs:
   - Step 1 / 3 — Reading the picture (image captioning, ~30–60 s)
   - Step 2 / 3 — Writing your story (LLM generation, ~60–120 s)
   - Step 3 / 3 — Recording the story (gTTS, ~5–10 s)

4. **Read and listen** to the generated story displayed in a styled card.
   Use the audio player to hear it read aloud.

5. **Download** the story text (`.txt`) or audio (`.mp3`) using the download
   buttons.

### Advanced options (sidebar expander)

| Option | Default | Description |
|---|---|---|
| Show image caption | ✅ On | Displays the raw caption produced by GIT-base-COCO |
| Show debug info | ❌ Off | Shows model names, word count, per-step timings, and selected style |

---

## 11. Configuration Reference

All application-level constants are defined at the top of `app.py`
(Section 1) and can be adjusted without modifying any logic:

| Constant | Default | Description |
|---|---|---|
| `TARGET_MIN_WORDS` | `50` | Stories below this count trigger a soft warning |
| `TARGET_MAX_WORDS` | `120` | Soft upper bound shown in the word-count warning |
| `TARGET_HARD_MAX` | `110` | Hard ceiling enforced by sentence-boundary truncation |
| `CAPTION_MODEL_NAME` | `"microsoft/git-base-coco"` | Hugging Face model ID for captioning |
| `STORY_MODEL_NAME` | `"Qwen/Qwen2.5-0.5B-Instruct"` | Hugging Face model ID for story generation |
| `TTS_LANGUAGE` | `"en"` | BCP-47 language tag passed to gTTS |
| `TTS_SLOW_SPEECH` | `False` | Set to `True` for slower TTS playback |
| `BANNED_TERMS` | *(list of 12 terms)* | Words that trigger the safety fallback |

---

## 12. Child-Safety Design

The application targets children aged **4–8** and implements a multi-layer
safety approach:

### Layer 1 — LLM system message constraints
The Qwen system prompt instructs the model to use only simple, everyday
vocabulary and specifies a child-appropriate tone and ending for each style.

### Layer 2 — Post-processing safety filter (`enforce_story_constraints`)
After generation, the story is scanned for 12 banned terms using
**whole-word regex matching** (`\b` word-boundary anchors). This prevents
false positives on substrings (e.g. the word *"dead"* will not match inside
*"instead"*).

If any banned term is detected, the entire story is replaced with a
guaranteed-safe fallback sentence:

> *"Once upon a time, a little friend went on a gentle adventure and came
> home happy, warm, and full of joy."*

### Layer 3 — Length constraints
Stories exceeding `TARGET_HARD_MAX` words are trimmed at the nearest sentence
boundary (never mid-sentence) to maintain readability and age-appropriateness.

---

## 13. Assessment Criteria Mapping

| Criterion | Implementation |
|---|---|
| **Functionality** — processes images, generates story, converts to audio | Three-step pipeline: `run_caption_model` → `run_story_model` → `run_tts`; all three outputs verified on every run |
| **Code Quality** — well-structured, documented | Single file with 7 labelled sections; module-level docstring with navigation guide |
| **Code Quality** — functions for modularity | 12 named functions each with a single responsibility; `main()` only orchestrates |
| **Code Quality** — meaningful variable names | Full-word names throughout (e.g. `caption_start_time`, `final_word_count`, `show_image_caption`); no single-letter or abbreviated names in logic |
| **Code Quality** — proper indentation | PEP 8 compliant; 4-space indentation consistently applied |
| **Code Quality** — code documentation | Google-style docstrings on all 12 functions (summary, Args, Returns, Raises); inline comments on every non-obvious line |
| **Model Usage** — appropriate Hugging Face models | `microsoft/git-base-coco` (image-to-text) and `Qwen/Qwen2.5-0.5B-Instruct` (text generation) selected after evaluating multiple alternatives for quality vs. RAM budget |
| **User Experience** — interactive, user-friendly Streamlit UI | Gradient banner, styled story card, three-step progress bar, style radio buttons, expander for advanced options, three-column results row, dashed upload hint area |
| **Deployment** — successfully deployed on Streamlit Cloud | Live at [https://individualassignment-wenjing106.streamlit.app](https://individualassignment-wenjing106.streamlit.app) |

---

## 14. Dependencies

| Package | Version constraint | Purpose |
|---|---|---|
| `streamlit` | `>=1.33,<2` | Web UI framework |
| `transformers` | `>=4.40,<5` | Hugging Face model loading and inference |
| `torch` | `>=2.1,<3` | PyTorch tensor operations and model weights |
| `Pillow` | `>=10,<12` | Image loading and RGB conversion |
| `gTTS` | `>=2.5,<3` | Google Text-to-Speech via HTTPS |
| `sentencepiece` | `>=0.2,<1` | Tokeniser dependency for Qwen models |
| `accelerate` | `>=0.30,<2` | Required by Transformers for `device_map="cpu"` |

---

## 15. Known Limitations

| Limitation | Details |
|---|---|
| **Slow first-run** | Model weights (~1.2 GB total) are downloaded on first access; Streamlit Cloud may show a blank screen for 2–4 minutes |
| **CPU-only inference** | No GPU acceleration; story generation takes approximately 60–120 seconds per run on the free tier |
| **Caption accuracy** | GIT-base-COCO performs best on photographic images similar to COCO training data; heavily stylised illustrations or abstract art may produce less accurate captions |
| **Story–image alignment** | Story quality is bounded by caption quality; if the caption misses key scene elements, the story may not fully reflect the image |
| **English only** | Both the story model prompt and gTTS are configured for English; other languages are not supported in the current version |
| **gTTS requires internet** | The TTS step makes an outbound HTTPS request to Google's API; the app cannot generate audio in a fully offline environment |

---

*Built for ISOM5240 Individual Assignment · Streamlit · GIT-base-COCO · Qwen2.5-0.5B-Instruct · gTTS*
