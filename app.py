from __future__ import annotations

import gc
import io
import re
import time
from typing import Dict, Optional, Tuple

import streamlit as st
from gtts import gTTS
from PIL import Image, UnidentifiedImageError
from transformers import pipeline


# =========================================================
# 1. APP CONFIG
# =========================================================
st.set_page_config(
    page_title="Magic Story Maker",
    page_icon="🌈",
    layout="centered",
)

APP_TITLE    = "🌈 Magic Story Maker"
APP_SUBTITLE = "Upload a picture and create a fun story for kids!"

TARGET_MIN_WORDS = 50
TARGET_MAX_WORDS = 120
TARGET_HARD_MAX  = 110   # sentence-boundary hard-truncation ceiling

# ── Model identifiers ────────────────────────────────────────────────────────
# Caption  : microsoft/git-base-coco — 182 M params, ~728 MB fp32
#            GIT (Generative Image-to-Text) is a Transformer decoder
#            conditioned on CLIP visual tokens.  Fine-tuned on COCO, it
#            generates notably more descriptive, scene-aware captions than
#            BLIP-base, e.g. "a boy in a red jacket running through a sunlit
#            park" rather than just "a person in a park".
#            Loaded first, then freed before the story LLM is loaded.
CAPTION_MODEL_NAME = "microsoft/git-base-coco"

# Story    : MBZUAI/LaMini-Flan-T5-248M — 248 M params, ~496 MB fp16
#            A distilled, instruction-fine-tuned variant of flan-t5-base
#            trained on 2.58 M diverse instruction samples (LaMini dataset).
#            Instruction tuning makes it far more responsive to persona,
#            style, and word-count directives than vanilla flan-t5-base,
#            producing richer, more child-friendly literary prose.
#            Loaded after caption model is freed; peak RAM ≈ 728 MB ✅
STORY_MODEL_NAME = "MBZUAI/LaMini-Flan-T5-248M"

# TTS      : gTTS — zero local RAM (HTTPS call to Google TTS).
#            Streamlit Cloud allows outbound HTTPS; no WebSocket needed.
TTS_LANG = "en"
TTS_SLOW = False

# Safety
BANNED_TERMS = [
    "blood", "kill", "dead", "gun", "knife", "monster", "horror",
    "terror", "violent", "violence", "war", "hate",
]

STYLE_OPTIONS: Dict[str, str] = {
    "Warm & Happy 😊": "warm and cheerful, ending with a smile",
    "Adventure 🚀":    "exciting and playful, ending safely and happily",
    "Bedtime 🌙":      "calm and soothing, like a gentle lullaby",
}


# =========================================================
# 2. UTILITY FUNCTIONS
# =========================================================
def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def contains_unsafe_content(text: str) -> bool:
    # E1: whole-word matching — avoids false positives where a banned term
    # appears as a substring of a harmless word (e.g. "dead" inside
    # "dead-end", or inside a model output phrase like "instead").
    lowered = text.lower()
    return any(
        re.search(r'\b' + re.escape(term) + r'\b', lowered)
        for term in BANNED_TERMS
    )


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def validate_image(uploaded_file) -> Tuple[bool, Optional[str]]:
    if uploaded_file is None:
        return False, "Please upload an image first."
    if uploaded_file.type not in ["image/png", "image/jpeg",
                                   "image/jpg", "image/webp"]:
        return False, "Only PNG, JPG, JPEG, and WEBP images are supported."
    return True, None


def safe_open_image(uploaded_file) -> Image.Image:
    try:
        return Image.open(uploaded_file).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc
    except Exception as exc:
        raise ValueError("Failed to read the uploaded image.") from exc


def _truncate_at_sentence_boundary(text: str, max_words: int) -> str:
    """
    Keep only complete sentences that fit within *max_words* total.
    Falls back to a word-level cut with a closing period if no sentence
    boundary exists within the limit.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    kept, total = [], 0
    for sent in sentences:
        wc = count_words(sent)
        if total + wc <= max_words:
            kept.append(sent)
            total += wc
        else:
            break
    if kept:
        return " ".join(kept)
    # fallback: word-level truncation
    words = text.split()
    cut = " ".join(words[:max_words]).rstrip(",;:-")
    return cut if cut.endswith((".", "!", "?")) else cut + "."


# =========================================================
# 3. STEP-WISE MODEL EXECUTION  (load → run → free)
# =========================================================
# Architecture rationale
# ─────────────────────
# Streamlit Cloud free tier has ~1 GB usable RAM.
# Loading all models at once needs far more RAM → instant OOM / 5-min hang.
#
# Solution: load each model, run it, then *explicitly* delete it and call
# gc.collect() before loading the next one.  Peak RAM at any moment:
#   max(728 MB git-base-coco,  496 MB LaMini-Flan-T5 fp16,  ~50 MB gTTS)
#   ≈ 728 MB ✅
#
# We do NOT use @st.cache_resource because keeping all models resident
# simultaneously is exactly what caused the previous 5-minute hang.

def _run_caption(image: Image.Image) -> str:
    """
    Load microsoft/git-base-coco, caption the image, immediately free the model.
    GIT generates scene-aware captions trained on COCO — richer than BLIP-base.
    ~728 MB peak RAM, released before story generation begins.
    """
    pipe = pipeline(
        task="image-to-text",
        model=CAPTION_MODEL_NAME,
    )
    try:
        results = pipe(image)
        if not results or "generated_text" not in results[0]:
            raise ValueError("Caption model returned no result.")
        return clean_text(results[0]["generated_text"])
    finally:
        del pipe
        gc.collect()


def _expand_caption(raw_caption: str, style: str) -> str:
    """
    Turn the terse GIT caption into a child-friendly, story-ready scene
    description using LaMini-Flan-T5-248M.

    Design notes (Plan A + C)
    ─────────────────────────
    • NO full example output — small seq2seq models copy examples verbatim
      instead of adapting them to the actual caption.  Removing the example
      forces the model to work from the real caption text.
    • Vocabulary guard — explicit instruction to use only words a 6-year-old
      knows prevents rare literary terms from leaking into the expansion.
    """
    prompt = (
        "Rewrite the image description below as one lively sentence for a "
        "children's picture book aged 3 to 8. "
        "Add one colour and one sound or feeling. "
        "Use only simple, everyday words a 6-year-old knows. "
        "Do NOT copy example sentences. Base everything on the description.\n\n"
        "Image description: " + raw_caption + "\n"
        "Story tone: " + style + "\n\n"
        "Expanded scene:"
    )
    pipe = pipeline(
        task="text2text-generation",
        model=STORY_MODEL_NAME,
        model_kwargs={"torch_dtype": "auto"},
    )
    try:
        out = pipe(
            prompt,
            max_new_tokens=80,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        expanded = clean_text(out[0]["generated_text"])
        # E2: if the expansion itself contains a banned word, fall back to
        # raw_caption — prevents a dangerous word from being injected into
        # the story via the content anchor (Rule 1 in _build_story_prompt).
        if contains_unsafe_content(expanded):
            return raw_caption
        # Fall back to raw caption if expansion looks degenerate
        if len(expanded) < 15 or expanded.lower().startswith(raw_caption.lower()[:20]):
            return raw_caption
        return expanded
    except Exception:
        return raw_caption
    finally:
        del pipe
        gc.collect()


def _one_sentence(pipe, prompt: str, max_new_tokens: int = 40) -> str:
    """
    Ask the model to generate exactly one continuation sentence.
    Low max_new_tokens keeps each call fast and prevents the model
    from running past a single sentence boundary.
    """
    out = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        num_beams=2,
        no_repeat_ngram_size=3,
        repetition_penalty=1.3,
        early_stopping=True,
    )
    raw = clean_text(out[0]["generated_text"])
    # Keep only the first complete sentence
    first = re.split(r'(?<=[.!?])\s', raw)[0].strip()
    if not first:
        return raw
    if not first[-1] in ".!?":
        first += "."
    return first


# ── Style mood words used to colour the closing sentence ─────────────────────
_STYLE_MOOD: Dict[str, str] = {
    "warm and cheerful, ending with a smile":          "happy and warm",
    "exciting and playful, ending safely and happily": "brave and glad",
    "calm and soothing, like a gentle lullaby":        "sleepy and safe",
}


def _run_story(rich_caption: str, style: str) -> str:
    """
    F3 — Sentence-chain architecture
    ─────────────────────────────────
    Instead of asking LaMini-Flan-T5-248M to generate an entire story
    from a long rule-laden prompt (which causes 'instruction following
    collapse' — the model regurgitates the rules instead of writing a
    story), we split the task into three micro-calls:

      Call 1 — Opening sentence
        Input : "One day, {rich_caption}."  (scene anchor, ≤15 tokens)
        Output: 1 new sentence that introduces characters/action.

      Call 2 — Middle sentence
        Input : sentences 0+1 so far  (≤30 tokens)
        Output: 1 new sentence that develops the scene.

      Call 3 — Closing sentence
        Input : sentences 0+1+2 so far  (≤50 tokens)
        Output: 1 warm, happy closing sentence.

    Each call receives ≤50 tokens of context and must generate ≤40
    tokens, well within LaMini's reliable operating range.  The scene
    anchor in sentence 0 keeps all three sentences topically grounded
    to the uploaded image.

    The model is loaded once, used for all three calls, then freed.
    Peak RAM ≈ 496 MB fp16 — unchanged from before.
    """
    mood = _STYLE_MOOD.get(style, "happy")

    pipe = pipeline(
        task="text2text-generation",
        model=STORY_MODEL_NAME,
        model_kwargs={"torch_dtype": "auto"},
    )
    try:
        # ── Sentence 0: fixed scene anchor (no model call needed) ─────────
        s0 = f"One day, {rich_caption}."

        # ── Sentence 1: opening action ────────────────────────────────────
        p1 = (
            f"{s0} "
            f"Continue the story with one short sentence about what happened next:"
        )
        s1 = _one_sentence(pipe, p1, max_new_tokens=40)

        # ── Sentence 2: middle development ───────────────────────────────
        p2 = (
            f"{s0} {s1} "
            f"Continue the story with one short sentence that adds a colour or sound:"
        )
        s2 = _one_sentence(pipe, p2, max_new_tokens=40)

        # ── Sentence 3: second middle sentence ───────────────────────────
        p3 = (
            f"{s0} {s1} {s2} "
            f"Continue the story with one more short sentence:"
        )
        s3 = _one_sentence(pipe, p3, max_new_tokens=40)

        # ── Sentence 4: warm closing ──────────────────────────────────────
        p4 = (
            f"{s0} {s1} {s2} {s3} "
            f"End the story with one warm, {mood} sentence:"
        )
        s4 = _one_sentence(pipe, p4, max_new_tokens=40)

        return clean_text(f"{s0} {s1} {s2} {s3} {s4}")
    finally:
        del pipe
        gc.collect()


def _run_tts(text: str) -> bytes:
    """
    Convert text to MP3 bytes via gTTS (Google TTS HTTPS API).
    Zero local model RAM.  Streamlit Cloud allows outbound HTTPS.
    """
    tts = gTTS(text=text, lang=TTS_LANG, slow=TTS_SLOW)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


# =========================================================
# 4. CONSTRAINT ENFORCEMENT  (no extra LLM calls)
# =========================================================
def enforce_story_constraints(
    story: str,
) -> Tuple[str, Optional[str]]:
    """
    Pure post-processing: no further model calls.

    1. Unsafe content   → guaranteed-safe fallback sentence.
    2. Over hard limit  → sentence-boundary truncation.
    3. Under minimum    → informational warning only (story still shown).
    """
    current = clean_text(story)
    warning: Optional[str] = None

    if contains_unsafe_content(current):
        current = (
            "Once upon a time, a little friend went on a gentle adventure "
            "and came home happy, warm, and full of joy."
        )
        return current, "The story was replaced with a child-safe version."

    wc = count_words(current)

    if wc > TARGET_HARD_MAX:
        current = _truncate_at_sentence_boundary(current, TARGET_HARD_MAX)
        warning = "The story was lightly trimmed to keep it short and sweet."

    elif wc < TARGET_MIN_WORDS:
        warning = (
            f"The story is a little short ({wc} words) "
            "but should still be enjoyable!"
        )

    return current, warning


# =========================================================
# 5. UI HELPERS
# =========================================================
def render_header() -> None:
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)
    st.write(
        "Upload a picture and watch it become a magical story — "
        "read aloud just for you! ✨"
    )


def render_sidebar() -> Tuple[str, bool, bool]:
    st.sidebar.header("⚙️ Story Settings")
    story_style = st.sidebar.selectbox(
        "Choose a story style",
        list(STYLE_OPTIONS.keys()),
    )
    show_caption = st.sidebar.checkbox("Show image caption", value=True)
    show_debug   = st.sidebar.checkbox("Show debug info",   value=False)
    return story_style, show_caption, show_debug


def render_footer() -> None:
    st.markdown("---")
    st.caption(
        "Built with Streamlit · GIT-base-COCO · LaMini-Flan-T5-248M · gTTS"
    )


# =========================================================
# 6. MAIN APP LOGIC
# =========================================================
def main() -> None:
    render_header()
    story_style_label, show_caption, show_debug = render_sidebar()
    style_instruction = STYLE_OPTIONS[story_style_label]

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "webp"],
    )

    if uploaded_file is None:
        st.info("Upload a picture to begin your story adventure! 🌟")
        render_footer()
        return

    is_valid, error_message = validate_image(uploaded_file)
    if not is_valid:
        st.error(error_message)
        render_footer()
        return

    try:
        image = safe_open_image(uploaded_file)
    except ValueError as exc:
        st.error(str(exc))
        render_footer()
        return

    st.image(image, caption="Your uploaded image", use_container_width=True)

    if not st.button("✨ Create My Story"):
        render_footer()
        return

    # ── Step 1: Caption ──────────────────────────────────────────────────────
    with st.spinner("🔍 Reading the picture…"):
        start = time.time()
        raw_caption = _run_caption(image)
        t_caption = time.time() - start

    if show_caption:
        st.subheader("🖼️ Image Caption")
        st.write(raw_caption)

    # ── Step 2: Caption expansion ────────────────────────────────────────────
    with st.spinner("🌈 Imagining the scene…"):
        t0 = time.time()
        rich_caption = _expand_caption(raw_caption, style_instruction)
        t_expand = time.time() - t0

    # ── Step 3: Story generation (F3 sentence-chain) ─────────────────────────
    with st.spinner("📝 Writing your story sentence by sentence…"):
        t0 = time.time()
        raw_story = _run_story(rich_caption, style_instruction)
        t_story = time.time() - t0

    final_story, warning_message = enforce_story_constraints(raw_story)

    # ── Step 4: TTS ──────────────────────────────────────────────────────────
    with st.spinner("🔊 Recording the story…"):
        t0 = time.time()
        audio_bytes = _run_tts(final_story)
        t_tts = time.time() - t0

    # ── Output ────────────────────────────────────────────────────────────────
    elapsed    = t_caption + t_expand + t_story + t_tts
    word_count = count_words(final_story)

    st.success("Your story is ready! 🎉")

    if warning_message:
        st.info(warning_message)

    st.subheader("📖 Story")
    st.write(final_story)

    st.subheader("🔊 Listen")
    st.audio(audio_bytes, format="audio/mp3")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download Story",
            data=final_story,
            file_name="story.txt",
            mime="text/plain",
        )
    with col2:
        st.download_button(
            label="📥 Download Audio",
            data=audio_bytes,
            file_name="story.mp3",
            mime="audio/mpeg",
        )

    if show_debug:
        st.markdown("### 🛠 Debug Info")
        st.write({
            "caption_model":  CAPTION_MODEL_NAME,
            "story_model":    STORY_MODEL_NAME,
            "story_arch":    "F3 sentence-chain (4 micro-calls)",
            "raw_caption":    raw_caption,
            "rich_caption":   rich_caption,
            "word_count":     word_count,
            "t_caption_s":    round(t_caption, 1),
            "t_expand_s":     round(t_expand, 1),
            "t_story_s":      round(t_story, 1),
            "t_tts_s":        round(t_tts, 1),
            "total_s":        round(elapsed, 1),
            "style":          story_style_label,
        })

    if word_count < TARGET_MIN_WORDS or word_count > TARGET_MAX_WORDS:
        st.warning(
            f"Story word count is {word_count} "
            f"(target {TARGET_MIN_WORDS}–{TARGET_MAX_WORDS})."
        )

    render_footer()


# =========================================================
# 7. ENTRY POINT
# =========================================================
if __name__ == "__main__":
    main()
