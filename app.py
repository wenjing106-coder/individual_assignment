from __future__ import annotations

import io
import re
import struct
import time
from typing import Dict, Optional, Tuple

import numpy as np
import streamlit as st
from kokoro import KPipeline
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

# ── Kokoro TTS ──────────────────────────────────────────
# Kokoro-82M: open-weight neural TTS, fully offline, no network needed.
# Runs on CPU via PyTorch. Models are cached by HF Hub after first download.
# Voice af_heart = warm American-English female voice ("heart" voice)
TTS_VOICE      = "af_heart"   # warm, nurturing storytelling voice
TTS_LANG_CODE  = "a"          # 'a' = American English
TTS_SPEED      = 0.92         # slightly slower for young listeners

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
    torch_dtype=auto  → float32 on CPU, bfloat16 if GPU found.
    low_cpu_mem_usage → loads layer-by-layer to keep peak RAM low.
    """
    tokenizer = AutoTokenizer.from_pretrained(STORY_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        STORY_MODEL_NAME,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    return tokenizer, model


@st.cache_resource(show_spinner="Loading voice model…")
def load_tts_pipeline() -> KPipeline:
    """
    Kokoro-82M KPipeline — loaded once and cached for the session.

    Kokoro is a fully offline neural TTS engine (82M parameters).
    It requires espeak-ng for English grapheme-to-phoneme conversion,
    which is declared in packages.txt for Streamlit Cloud.
    Voice 'af_heart' is the warm, nurturing American-English female voice.
    """
    return KPipeline(lang_code=TTS_LANG_CODE)


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


# ── 4d. Text-to-Speech via Kokoro-82M ───────────────────
def _numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int = 24000) -> bytes:
    """
    Convert a float32 numpy audio array to an in-memory WAV file (bytes).

    Kokoro outputs float32 PCM at 24 kHz. We normalise, convert to int16,
    then wrap in a minimal WAV header so Streamlit's st.audio() can play it.
    """
    # Normalise to [-1, 1] and convert to 16-bit PCM
    audio_clipped = np.clip(audio, -1.0, 1.0)
    pcm = (audio_clipped * 32767).astype(np.int16)
    pcm_bytes = pcm.tobytes()

    # Build a standard WAV header (44 bytes)
    num_channels    = 1
    bits_per_sample = 16
    byte_rate       = sample_rate * num_channels * bits_per_sample // 8
    block_align     = num_channels * bits_per_sample // 8
    data_size       = len(pcm_bytes)
    chunk_size      = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", chunk_size, b"WAVE",
        b"fmt ", 16,
        1,               # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data", data_size,
    )
    return header + pcm_bytes


def text_to_speech_bytes(text: str) -> bytes:
    """
    Convert story text to WAV bytes using Kokoro-82M neural TTS.

    Kokoro runs fully offline — no WebSocket, no external API.
    The 'af_heart' voice is warm and nurturing, ideal for children's
    stories. Audio chunks are concatenated then encoded as WAV.
    """
    kokoro = load_tts_pipeline()
    chunks = []
    for _, _, audio in kokoro(text, voice=TTS_VOICE, speed=TTS_SPEED):
        if audio is not None and len(audio) > 0:
            chunks.append(audio)

    if not chunks:
        raise RuntimeError("Kokoro TTS returned no audio for the given text.")

    full_audio = np.concatenate(chunks)
    return _numpy_to_wav_bytes(full_audio)


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
        "Qwen2.5-0.5B-Instruct · Kokoro-82M Neural TTS"
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

                    # Step 4 – Warm TTS (Kokoro-82M, fully offline)
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
                st.audio(audio_bytes, format="audio/wav")

                st.download_button(
                    label="📥 Download Story as Text",
                    data=final_story,
                    file_name="story.txt",
                    mime="text/plain",
                )
                st.download_button(
                    label="📥 Download Audio",
                    data=audio_bytes,
                    file_name="story.wav",
                    mime="audio/wav",
                )

                if show_debug:
                    st.markdown("### 🛠 Debug Info")
                    st.write({
                        "caption_model": CAPTION_MODEL_NAME,
                        "story_model":   STORY_MODEL_NAME,
                        "tts_engine":    "Kokoro-82M",
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
