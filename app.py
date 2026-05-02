from __future__ import annotations

import gc
import io
import re
import time
from typing import Dict, Optional, Tuple

import streamlit as st
from gtts import gTTS
from PIL import Image, UnidentifiedImageError
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


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
#            GIT (Generative Image-to-Text) decoder conditioned on CLIP
#            visual tokens; fine-tuned on COCO.  Generates scene-aware
#            captions such as "a boy in a red jacket running through a
#            sunlit park".  Loaded first, freed before story LLM loads.
CAPTION_MODEL_NAME = "microsoft/git-base-coco"

# Story    : Qwen/Qwen2.5-0.5B-Instruct — 494 M params, ~500 MB fp16
#            Decoder-only causal LM with instruction fine-tuning.
#            Unlike seq2seq models (T5, LaMini), a decoder LM never
#            "regurgitates" prompt rules into the output — it simply
#            continues the conversation.  The chat-template interface
#            separates system/user/assistant roles cleanly, so the model
#            receives a concise scene description and returns a story.
#            Loaded after caption model freed; peak RAM ≈ 728 MB ✅
STORY_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

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
    "Warm & Happy 😊": (
        "warm and cheerful",
        "The story ends with everyone smiling and feeling happy."
    ),
    "Adventure 🚀": (
        "exciting and playful",
        "The story ends with a safe, joyful adventure completed."
    ),
    "Bedtime 🌙": (
        "calm and soothing",
        "The story ends peacefully, like a gentle lullaby sending the characters to sleep."
    ),
}


# =========================================================
# 2. UTILITY FUNCTIONS
# =========================================================
def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def contains_unsafe_content(text: str) -> bool:
    # Whole-word matching — avoids false positives on substrings
    # (e.g. "dead" inside "instead").
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
# Memory budget on Streamlit Cloud free tier (~1 GB usable RAM)
# ──────────────────────────────────────────────────────────────
# Models are loaded sequentially and freed immediately after use:
#
#   Step 1  git-base-coco (caption)    ~728 MB  → del + gc
#   Step 2  Qwen2.5-0.5B-Instruct fp16 ~500 MB  → del + gc
#   Step 3  gTTS (HTTPS, no local RAM)   ~0 MB
#
#   Peak RAM = max(728, 500) = 728 MB  ✅  (well under 1 GB)

def _run_caption(image: Image.Image) -> str:
    """
    Load microsoft/git-base-coco, caption the image, free the model.
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


def _build_chat_messages(caption: str, style_tone: str,
                          style_ending: str) -> list:
    """
    Build the chat-template message list for Qwen2.5-0.5B-Instruct.

    G2 design rationale
    ───────────────────
    Qwen2.5-0.5B-Instruct is a decoder-only causal LM with instruction
    fine-tuning.  Its chat template separates system / user / assistant
    roles so the model never confuses "instructions" with "story output".

    The system message establishes a children's author persona with clear
    vocabulary and length constraints.  The user message provides only
    the scene — a short, focused input that leaves no room for the model
    to leak prompt text back.

    Crucially we do NOT embed style descriptions or word-count rules as
    numbered rules inside the user turn — doing so caused LaMini to copy
    the rule text verbatim.  Instead, all structural constraints live in
    the system message, which Qwen treats as background context rather
    than content to reproduce.
    """
    system_msg = (
        "You are a kind and imaginative children's storyteller. "
        "When given a scene description, you write a short, original story "
        "for children aged 4 to 8. "
        "Always use simple, everyday words a young child understands. "
        "Your stories are " + style_tone + ". "
        + style_ending + " "
        "Write between 60 and 90 words. "
        "Do not include a title. Do not repeat sentences."
    )
    user_msg = (
        "Write a children's story about this scene:\n"
        + caption
    )
    return [
        {"role": "system",  "content": system_msg},
        {"role": "user",    "content": user_msg},
    ]


def _run_story(caption: str, style_label: str) -> str:
    """
    G2 — Qwen2.5-0.5B-Instruct with chat template
    ───────────────────────────────────────────────
    Load Qwen2.5-0.5B-Instruct in fp16, generate the story via the
    official chat-template interface, then free the model.

    Why Qwen2.5-0.5B-Instruct outperforms LaMini / flan-t5:
    • Decoder-only architecture: the model appends to the assistant turn
      rather than transforming input text, so prompt rules never appear
      in the output.
    • Instruction fine-tuning on diverse creative tasks: the model
      genuinely understands "write a children's story" and honours
      length / style constraints without repeating them.
    • fp16 weight loading keeps peak RAM at ~500 MB — safely within the
      728 MB already occupied by the caption step.

    Generation parameters chosen for story quality:
    • do_sample=True, temperature=0.8  — controlled creativity without
      wild hallucinations; deterministic beam search on a small model
      tends to produce repetitive high-probability phrases.
    • top_p=0.9                        — nucleus sampling filters the
      very long tail of low-probability tokens.
    • repetition_penalty=1.15          — mild penalty keeps successive
      sentences from starting with the same phrase.
    • max_new_tokens=180               — generous ceiling; constraint
      enforcement trims if needed.
    • min_new_tokens=60                — prevents a one-sentence output.
    """
    style_tone, style_ending = STYLE_OPTIONS[style_label]
    messages = _build_chat_messages(caption, style_tone, style_ending)

    tokenizer = AutoTokenizer.from_pretrained(STORY_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        STORY_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    try:
        # Apply Qwen's built-in chat template to format the messages
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=180,
                min_new_tokens=60,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (strip the prompt)
        new_tokens = output_ids[0][input_len:]
        story = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return clean_text(story)
    finally:
        del model, tokenizer
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
def enforce_story_constraints(story: str) -> Tuple[str, Optional[str]]:
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
        "Built with Streamlit · GIT-base-COCO · Qwen2.5-0.5B-Instruct · gTTS"
    )


# =========================================================
# 6. MAIN APP LOGIC
# =========================================================
def main() -> None:
    render_header()
    story_style_label, show_caption, show_debug = render_sidebar()

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

    # ── Step 2: Story generation (G2 — Qwen2.5-0.5B-Instruct) ───────────────
    with st.spinner("📝 Writing your story…"):
        t0 = time.time()
        raw_story = _run_story(raw_caption, story_style_label)
        t_story = time.time() - t0

    final_story, warning_message = enforce_story_constraints(raw_story)

    # ── Step 3: TTS ──────────────────────────────────────────────────────────
    with st.spinner("🔊 Recording the story…"):
        t0 = time.time()
        audio_bytes = _run_tts(final_story)
        t_tts = time.time() - t0

    # ── Output ────────────────────────────────────────────────────────────────
    elapsed    = t_caption + t_story + t_tts
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
            "caption_model": CAPTION_MODEL_NAME,
            "story_model":   STORY_MODEL_NAME,
            "story_arch":    "G2 chat-template (single call)",
            "raw_caption":   raw_caption,
            "word_count":    word_count,
            "t_caption_s":   round(t_caption, 1),
            "t_story_s":     round(t_story, 1),
            "t_tts_s":       round(t_tts, 1),
            "total_s":       round(elapsed, 1),
            "style":         story_style_label,
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
