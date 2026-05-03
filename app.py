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
CAPTION_MODEL_NAME = "microsoft/git-base-coco"
STORY_MODEL_NAME   = "Qwen/Qwen2.5-0.5B-Instruct"
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

# ── Gradient colours per style ───────────────────────────────────────────────
STYLE_GRADIENTS: Dict[str, str] = {
    "Warm & Happy 😊": "linear-gradient(135deg, #FFECD2 0%, #FCB69F 100%)",
    "Adventure 🚀":    "linear-gradient(135deg, #A1C4FD 0%, #C2E9FB 100%)",
    "Bedtime 🌙":      "linear-gradient(135deg, #D4B3F5 0%, #8EC5FC 100%)",
}


# =========================================================
# 2. UTILITY FUNCTIONS
# =========================================================
def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def contains_unsafe_content(text: str) -> bool:
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
    words = text.split()
    cut = " ".join(words[:max_words]).rstrip(",;:-")
    return cut if cut.endswith((".", "!", "?")) else cut + "."


# =========================================================
# 3. STEP-WISE MODEL EXECUTION  (load → run → free)
# =========================================================
def _run_caption(image: Image.Image) -> str:
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
    style_tone, style_ending = STYLE_OPTIONS[style_label]
    messages = _build_chat_messages(caption, style_tone, style_ending)

    tokenizer = AutoTokenizer.from_pretrained(STORY_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        STORY_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    try:
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

        new_tokens = output_ids[0][input_len:]
        story = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return clean_text(story)
    finally:
        del model, tokenizer
        gc.collect()


def _run_tts(text: str) -> bytes:
    tts = gTTS(text=text, lang=TTS_LANG, slow=TTS_SLOW)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


# =========================================================
# 4. CONSTRAINT ENFORCEMENT
# =========================================================
def enforce_story_constraints(story: str) -> Tuple[str, Optional[str]]:
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

# ── U1: Gradient banner ───────────────────────────────────────────────────────
def render_header(style_label: str = "Warm & Happy 😊") -> None:
    gradient = STYLE_GRADIENTS.get(style_label, STYLE_GRADIENTS["Warm & Happy 😊"])
    st.markdown(
        f"""
        <div style="
            background: {gradient};
            border-radius: 16px;
            padding: 2rem 2.5rem 1.5rem 2.5rem;
            margin-bottom: 1.5rem;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        ">
            <div style="font-size: 3rem; margin-bottom: 0.3rem;">🌈</div>
            <h1 style="
                margin: 0;
                font-size: 2.2rem;
                font-weight: 800;
                color: #2d2d2d;
                letter-spacing: -0.5px;
            ">Magic Story Maker</h1>
            <p style="
                margin: 0.5rem 0 0 0;
                font-size: 1.05rem;
                color: #555;
            ">Upload a picture and create a fun story for kids! ✨</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── U2: Story card ────────────────────────────────────────────────────────────
def render_story_card(story: str, style_label: str) -> None:
    gradient = STYLE_GRADIENTS.get(style_label, STYLE_GRADIENTS["Warm & Happy 😊"])
    st.markdown(
        f"""
        <div style="
            background: {gradient};
            border-radius: 14px;
            border-left: 6px solid rgba(0,0,0,0.12);
            padding: 1.4rem 1.8rem;
            margin: 0.5rem 0 1.2rem 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.07);
            font-size: 1.08rem;
            line-height: 1.75;
            color: #2d2d2d;
        ">
            {story}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── U4: Sidebar redesign ──────────────────────────────────────────────────────
def render_sidebar() -> Tuple[str, bool, bool]:
    st.sidebar.markdown(
        """
        <div style="
            background: linear-gradient(135deg,#FFECD2,#FCB69F);
            border-radius: 12px;
            padding: 0.8rem 1rem;
            margin-bottom: 1rem;
            text-align: center;
        ">
            <span style="font-size:1.4rem;">⚙️</span>
            <span style="font-weight:700; font-size:1rem; color:#2d2d2d;">
              &nbsp;Story Settings
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Radio buttons for style selection (U4)
    story_style = st.sidebar.radio(
        "🎨 Story Style",
        list(STYLE_OPTIONS.keys()),
        horizontal=False,
        index=0,
    )

    st.sidebar.markdown("---")

    # Expander keeps advanced options out of the way (U4)
    with st.sidebar.expander("🔧 Advanced Options", expanded=False):
        show_caption = st.checkbox(
            "Show image caption", value=True, key="show_caption_cb"
        )
        show_debug = st.checkbox(
            "Show debug info", value=False, key="show_debug_cb"
        )

    return story_style, show_caption, show_debug


def render_footer() -> None:
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#888; font-size:0.82rem;'>"
        "Built with ❤️ using "
        "<b>Streamlit</b> · <b>GIT-base-COCO</b> · "
        "<b>Qwen2.5-0.5B-Instruct</b> · <b>gTTS</b>"
        "</p>",
        unsafe_allow_html=True,
    )


# =========================================================
# 6. MAIN APP LOGIC
# =========================================================
def main() -> None:
    # We need style early for the gradient banner, but sidebar must render first
    # so we render sidebar, capture style, then render header.
    story_style_label, show_caption, show_debug = render_sidebar()
    render_header(story_style_label)

    # ── U6: Upload area beautification ───────────────────────────────────────
    st.markdown(
        """
        <div style="
            background: #f9f9f9;
            border: 2px dashed #d0d0d0;
            border-radius: 14px;
            padding: 1.2rem 1.5rem 0.8rem 1.5rem;
            margin-bottom: 0.5rem;
            text-align: center;
        ">
            <div style="font-size: 2rem;">🖼️</div>
            <p style="margin: 0.3rem 0 0 0; color: #666; font-size: 0.95rem;">
                Drag &amp; drop your picture here, or click the button below
                <br><span style="color:#aaa; font-size:0.85rem;">
                    PNG · JPG · WEBP supported
                </span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.info("📸 Upload a picture above to begin your story adventure! 🌟")
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

    if not st.button("✨ Create My Story", use_container_width=True, type="primary"):
        render_footer()
        return

    # ── U3: Step progress indicator ──────────────────────────────────────────
    progress_bar  = st.progress(0, text="Starting…")
    status_text   = st.empty()

    # ── Step 1: Caption ──────────────────────────────────────────────────────
    status_text.markdown(
        "**Step 1 / 3** &nbsp;🔍&nbsp; Reading the picture…",
        unsafe_allow_html=True,
    )
    progress_bar.progress(5, text="Step 1 / 3 — Reading the picture…")

    start = time.time()
    raw_caption = _run_caption(image)
    t_caption = time.time() - start

    progress_bar.progress(35, text="Step 1 / 3 — Done ✅")

    if show_caption:
        with st.expander("🖼️ Image Caption", expanded=True):
            st.write(raw_caption)

    # ── Step 2: Story generation ─────────────────────────────────────────────
    status_text.markdown(
        "**Step 2 / 3** &nbsp;📝&nbsp; Writing your story…",
        unsafe_allow_html=True,
    )
    progress_bar.progress(40, text="Step 2 / 3 — Writing your story…")

    t0 = time.time()
    raw_story = _run_story(raw_caption, story_style_label)
    t_story = time.time() - t0

    final_story, warning_message = enforce_story_constraints(raw_story)
    progress_bar.progress(75, text="Step 2 / 3 — Done ✅")

    # ── Step 3: TTS ──────────────────────────────────────────────────────────
    status_text.markdown(
        "**Step 3 / 3** &nbsp;🔊&nbsp; Recording the story…",
        unsafe_allow_html=True,
    )
    progress_bar.progress(80, text="Step 3 / 3 — Recording the story…")

    t0 = time.time()
    audio_bytes = _run_tts(final_story)
    t_tts = time.time() - t0

    progress_bar.progress(100, text="All done! 🎉")
    status_text.empty()   # clear the step label once finished

    # ── Output ────────────────────────────────────────────────────────────────
    elapsed    = t_caption + t_story + t_tts
    word_count = count_words(final_story)

    st.success(f"Your story is ready! 🎉 ({word_count} words, {elapsed:.0f} s)")

    if warning_message:
        st.info(warning_message)

    # U2: Story displayed in a styled card
    st.markdown("### 📖 Your Story")
    render_story_card(final_story, story_style_label)

    # ── U5: Audio + downloads in a single row (3 columns) ────────────────────
    st.markdown("### 🔊 Listen & Download")
    col_audio, col_dl_txt, col_dl_mp3 = st.columns([3, 1, 1])

    with col_audio:
        st.audio(audio_bytes, format="audio/mp3")

    with col_dl_txt:
        st.download_button(
            label="📄 Story",
            data=final_story,
            file_name="story.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with col_dl_mp3:
        st.download_button(
            label="🎵 Audio",
            data=audio_bytes,
            file_name="story.mp3",
            mime="audio/mpeg",
            use_container_width=True,
        )

    # ── Debug info ───────────────────────────────────────────────────────────
    if show_debug:
        with st.expander("🛠 Debug Info", expanded=False):
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
