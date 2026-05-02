from __future__ import annotations

import asyncio
import io
import re
import tempfile
import time
from typing import Dict, Optional, Tuple

import edge_tts
import streamlit as st
from PIL import Image, UnidentifiedImageError
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)


# =========================================================
# 1. APP CONFIG
# =========================================================
st.set_page_config(
    page_title="Magic Story Maker",
    page_icon="🌈",
    layout="centered",
)

# ---------------------------------------------------------
# App-level constants
# ---------------------------------------------------------
APP_TITLE = "🌈 Magic Story Maker"
APP_SUBTITLE = "Upload a picture and create a fun story for kids!"

TARGET_MIN_WORDS = 50
TARGET_MAX_WORDS = 100

# ── Model identifiers ────────────────────────────────────
# Image captioning: GIT-Large fine-tuned on COCO
# Produces richer, more descriptive captions than BLIP-base
CAPTION_MODEL_NAME = "microsoft/git-large-coco"

# Story LLM: Qwen2.5-0.5B-Instruct
# ~1 GB RAM on CPU, instruction-tuned, strong creative writing
STORY_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# ── Edge-TTS voice ───────────────────────────────────────
# Microsoft Neural TTS – warm, nurturing English female voice
# Full voice list: run `edge-tts --list-voices` in a terminal
TTS_VOICE = "en-US-JennyNeural"          # warm, storytelling voice
TTS_RATE  = "-5%"                        # slightly slower for children
TTS_PITCH = "+2Hz"                       # gentle lift for warmth

# Safety / content control
BANNED_TERMS = [
    "blood", "kill", "dead", "gun", "knife", "monster", "horror",
    "terror", "violent", "violence", "war", "hate",
]

STYLE_OPTIONS = {
    "Warm & Happy 😊": (
        "warm, happy, and gentle with a cheerful ending"
    ),
    "Adventure 🚀": (
        "playful and exciting with a safe and happy ending"
    ),
    "Bedtime 🌙": (
        "calm, cozy, and soothing — perfect for falling asleep"
    ),
}


# =========================================================
# 2. UTILITY FUNCTIONS
# =========================================================
def count_words(text: str) -> int:
    """Return approximate English word count."""
    return len(re.findall(r"\b[\w'-]+\b", text))


def contains_unsafe_content(text: str) -> bool:
    """Simple keyword-based safety check."""
    lowered = text.lower()
    return any(term in lowered for term in BANNED_TERMS)


def clean_text(text: str) -> str:
    """Basic text cleanup."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def validate_image(uploaded_file) -> Tuple[bool, Optional[str]]:
    """Validate uploaded image file."""
    if uploaded_file is None:
        return False, "Please upload an image first."
    allowed_types = ["image/png", "image/jpeg", "image/jpg", "image/webp"]
    if uploaded_file.type not in allowed_types:
        return False, "Only PNG, JPG, JPEG, and WEBP images are supported."
    return True, None


def safe_open_image(uploaded_file) -> Image.Image:
    """Open uploaded image safely and convert to RGB."""
    try:
        return Image.open(uploaded_file).convert("RGB")
    except UnidentifiedImageError as e:
        raise ValueError("The uploaded file is not a valid image.") from e
    except Exception as e:
        raise ValueError("Failed to read the uploaded image.") from e


# =========================================================
# 3. MODEL LOADING (CACHED)
# =========================================================
@st.cache_resource(show_spinner="Loading image captioning model…")
def load_caption_pipeline():
    """
    GIT-Large fine-tuned on COCO.
    Produces richer, more descriptive captions than BLIP-base,
    which in turn gives the story model better raw material.
    """
    return pipeline(
        task="image-to-text",
        model=CAPTION_MODEL_NAME,
    )


@st.cache_resource(show_spinner="Loading story generation model…")
def load_story_model():
    """
    Qwen2.5-0.5B-Instruct loaded for CPU inference.
    torch_dtype=float32  → avoids bfloat16 issues on CPU-only hosts.
    low_cpu_mem_usage    → loads layer-by-layer to keep peak RAM low.
    """
    tokenizer = AutoTokenizer.from_pretrained(STORY_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        STORY_MODEL_NAME,
        torch_dtype="auto",        # float32 on CPU, bfloat16 if GPU found
        low_cpu_mem_usage=True,
    )
    return tokenizer, model


# =========================================================
# 4. CORE GENERATION FUNCTIONS
# =========================================================

# ── 4a. Image captioning ─────────────────────────────────
def generate_caption(image: Image.Image) -> str:
    """
    Generate an image caption using GIT-Large (COCO).

    GIT produces more detailed, context-aware descriptions than
    BLIP-base, giving the story model richer raw material.
    Example output:
        "a little girl in a yellow raincoat jumping in puddles
         on a rainy day, smiling and holding a red umbrella"
    """
    caption_pipe = load_caption_pipeline()
    results = caption_pipe(image)

    if not results or "generated_text" not in results[0]:
        raise ValueError("Caption model did not return a valid result.")

    return clean_text(results[0]["generated_text"])


# ── 4b. Story generation with Qwen2.5-0.5B-Instruct ──────
def _build_system_prompt() -> str:
    return (
        "You are a gifted children's author who writes in a lyrical, "
        "imaginative style — like a blend of Beatrix Potter and A.A. Milne. "
        "Your stories are warm, whimsical, and full of sensory detail. "
        "You always write in complete sentences and finish with a gentle, "
        "happy ending."
    )


def _build_user_prompt(caption: str, style: str) -> str:
    return (
        f"Write a children's story inspired by this scene:\n"
        f"\"{caption}\"\n\n"
        f"Requirements:\n"
        f"- Style: {style}\n"
        f"- Length: between 60 and 80 words — no more, no less.\n"
        f"- Language: simple, musical English for children aged 3–10.\n"
        f"- Tone: whimsical, warm, and imaginative.\n"
        f"- Include at least one vivid sensory detail (colour, sound, smell, "
        f"texture, or taste).\n"
        f"- End with a cosy, happy sentence.\n"
        f"- Output the story text ONLY — no title, no labels, no extra words."
    )


def generate_story(caption: str, style: str) -> str:
    """
    Generate a children's story via Qwen2.5-0.5B-Instruct.

    Uses the model's native chat template so instruction-following
    is applied correctly, producing noticeably richer prose than
    the seq2seq FLAN-T5 approach.
    """
    tokenizer, model = load_story_model()

    messages = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user",   "content": _build_user_prompt(caption, style)},
    ]

    # apply_chat_template formats the conversation correctly for Qwen
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt")

    # Generate — constrained to ~120 new tokens (~80 words)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=150,
        min_new_tokens=60,
        do_sample=True,
        temperature=0.85,      # creative but controlled
        top_p=0.92,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Strip the prompt tokens; keep only the newly generated part
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    story = tokenizer.decode(new_ids, skip_special_tokens=True)
    return clean_text(story)


# ── 4c. Constraint enforcement (unchanged logic) ─────────
def get_story_quality_flags(story: str) -> Dict[str, object]:
    wc = count_words(story)
    return {
        "word_count": wc,
        "unsafe":     contains_unsafe_content(story),
        "in_range":   TARGET_MIN_WORDS <= wc <= TARGET_MAX_WORDS,
    }


def _fix_story(story: str, caption: str, style: str, reason: str) -> str:
    """Single-pass rewrite with a corrective instruction prepended."""
    fix_caption = (
        f"{reason} "
        f"Rewrite the story to be between 60 and 80 words, "
        f"safe, warm, and child-friendly. "
        f"Keep the same scene: \"{caption}\". Style: {style}.\n\n"
        f"Original story:\n{story}"
    )
    return generate_story(fix_caption, style)


def enforce_story_constraints(
    story: str,
    caption: str,
    style: str,
    max_retries: int = 3,
) -> Tuple[str, Optional[str]]:
    """
    Enforce word-count (50–100) and safety constraints.
    Returns (final_story, optional_warning_message).
    """
    warning = None
    current = clean_text(story)

    for _ in range(max_retries):
        info = get_story_quality_flags(current)
        if info["in_range"] and not info["unsafe"]:
            return current, warning

        if info["unsafe"]:
            current = _fix_story(
                current, caption, style,
                "The story contains content unsuitable for children."
            )
            warning = "The story was rewritten to make it safer for children."

        elif info["word_count"] < TARGET_MIN_WORDS:
            current = _fix_story(
                current, caption, style,
                f"The story is too short ({info['word_count']} words)."
            )
            warning = "The story was expanded to meet the word limit."

        elif info["word_count"] > TARGET_MAX_WORDS:
            current = _fix_story(
                current, caption, style,
                f"The story is too long ({info['word_count']} words)."
            )
            warning = "The story was shortened to meet the word limit."

    final_info = get_story_quality_flags(current)
    if not final_info["in_range"] or final_info["unsafe"]:
        raise ValueError(
            f"Could not generate a valid story after {max_retries} retries. "
            f"Final word count: {final_info['word_count']}"
        )
    return current, warning


# ── 4d. Text-to-Speech via edge-tts ──────────────────────
def text_to_speech_bytes(text: str) -> bytes:
    """
    Convert story text to MP3 bytes using Microsoft Edge Neural TTS.

    edge-tts is async; we run it in a temporary file to avoid
    buffering issues with the streaming API, then read it back.
    The JennyNeural voice is warm and nurturing — ideal for
    children's bedtime or story-time content.
    """
    async def _synthesise(text: str, tmp_path: str) -> None:
        communicate = edge_tts.Communicate(
            text,
            voice=TTS_VOICE,
            rate=TTS_RATE,
            pitch=TTS_PITCH,
        )
        await communicate.save(tmp_path)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    # Run the async synthesis in a fresh event loop
    asyncio.run(_synthesise(text, tmp_path))

    with open(tmp_path, "rb") as f:
        return f.read()


# =========================================================
# 5. UI HELPERS
# =========================================================
def render_header():
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)
    st.write(
        "Upload a picture and watch it become a magical story — "
        "read aloud just for you! ✨"
    )


def render_sidebar():
    st.sidebar.header("⚙️ Story Settings")
    story_style = st.sidebar.selectbox(
        "Choose a story style",
        list(STYLE_OPTIONS.keys()),
    )
    show_caption = st.sidebar.checkbox("Show image caption", value=True)
    show_debug   = st.sidebar.checkbox("Show debug info",   value=False)
    return story_style, show_caption, show_debug


def render_footer():
    st.markdown("---")
    st.caption(
        "Built with Streamlit · GIT-Large (COCO) · "
        "Qwen2.5-0.5B-Instruct · Microsoft Edge Neural TTS"
    )


# =========================================================
# 6. MAIN APP LOGIC
# =========================================================
def main():
    render_header()
    story_style_label, show_caption, show_debug = render_sidebar()
    style_instruction = STYLE_OPTIONS[story_style_label]

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "webp"],
    )

    if uploaded_file is not None:
        is_valid, error_message = validate_image(uploaded_file)
        if not is_valid:
            st.error(error_message)
            render_footer()
            return

        try:
            image = safe_open_image(uploaded_file)
            st.image(image, caption="Your uploaded image", use_container_width=True)

            generate_btn = st.button("✨ Create My Story")

            if generate_btn:
                with st.spinner("Creating your magical story… 🪄"):
                    start_time = time.time()

                    # Step 1 – Rich image caption (GIT-Large)
                    caption = generate_caption(image)

                    # Step 2 – Story (Qwen2.5-0.5B-Instruct)
                    raw_story = generate_story(caption, style_instruction)

                    # Step 3 – Constraint enforcement
                    final_story, warning_message = enforce_story_constraints(
                        story=raw_story,
                        caption=caption,
                        style=style_instruction,
                    )

                    # Step 4 – Warm TTS (Edge Neural)
                    audio_bytes = text_to_speech_bytes(final_story)

                    elapsed    = time.time() - start_time
                    word_count = count_words(final_story)

                st.success("Your story is ready! 🎉")

                if warning_message:
                    st.info(warning_message)

                if show_caption:
                    st.subheader("🖼️ Image Caption")
                    st.write(caption)

                st.subheader("📖 Story")
                st.write(final_story)

                st.subheader("🔊 Listen")
                st.audio(audio_bytes, format="audio/mp3")

                st.download_button(
                    label="📥 Download Story as Text",
                    data=final_story,
                    file_name="story.txt",
                    mime="text/plain",
                )
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
                        "tts_voice":     TTS_VOICE,
                        "word_count":    word_count,
                        "elapsed_sec":   round(elapsed, 2),
                        "style":         story_style_label,
                    })

                if word_count < TARGET_MIN_WORDS or word_count > TARGET_MAX_WORDS:
                    st.warning(
                        f"Story word count is {word_count}, outside the preferred "
                        f"range ({TARGET_MIN_WORDS}–{TARGET_MAX_WORDS})."
                    )

        except Exception as e:
            st.error("Something went wrong while generating the story.")
            st.exception(e)

    else:
        st.info("Upload a picture to begin your story adventure! 🌟")

    render_footer()


# =========================================================
# 7. ENTRY POINT
# =========================================================
if __name__ == "__main__":
    main()
