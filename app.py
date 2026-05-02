from __future__ import annotations

import io
import re
import time
from typing import Optional, Tuple

import streamlit as st
from PIL import Image, UnidentifiedImageError
from transformers import pipeline
from gtts import gTTS


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

# Suggested models
CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
STORY_MODEL_NAME = "google/flan-t5-base"

# Safety / content control
BANNED_TERMS = [
    "blood", "kill", "dead", "gun", "knife", "monster", "horror",
    "terror", "violent", "violence", "war", "hate"
]

STYLE_OPTIONS = {
    "Warm & Happy 😊": "Write a warm, happy, gentle story with a cheerful ending.",
    "Adventure 🚀": "Write a playful, exciting adventure story with a safe and happy ending.",
    "Bedtime 🌙": "Write a calm, cozy bedtime story with soft and soothing language."
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


def truncate_to_max_words(text: str, max_words: int = TARGET_MAX_WORDS) -> str:
    """Trim text to max word limit if needed."""
    words = text.split()
    if len(words) <= max_words:
        return text
    trimmed = " ".join(words[:max_words]).rstrip(",;:-")
    if not trimmed.endswith((".", "!", "?")):
        trimmed += "."
    return trimmed


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
        image = Image.open(uploaded_file).convert("RGB")
        return image
    except UnidentifiedImageError as e:
        raise ValueError("The uploaded file is not a valid image.") from e
    except Exception as e:
        raise ValueError("Failed to read the uploaded image.") from e


# =========================================================
# 3. MODEL LOADING (CACHED)
# =========================================================
@st.cache_resource(show_spinner=True)
def load_caption_pipeline():
    """
    Load image captioning pipeline once.
    Using Hugging Face pipeline for assignment alignment.
    """
    return pipeline(
        task="image-to-text",
        model=CAPTION_MODEL_NAME
    )


@st.cache_resource(show_spinner=True)
def load_story_pipeline():
    """
    Load story generation pipeline once.
    FLAN-T5 works well with instruction-style prompts.
    """
    return pipeline(
        task="text2text-generation",
        model=STORY_MODEL_NAME
    )


# =========================================================
# 4. CORE GENERATION FUNCTIONS
# =========================================================
def generate_caption(image: Image.Image) -> str:
    """Generate an image caption from uploaded image."""
    caption_pipe = load_caption_pipeline()
    results = caption_pipe(image)

    # Expected HF output: list of dicts, e.g. [{"generated_text": "..."}]
    if not results or "generated_text" not in results[0]:
        raise ValueError("Caption model did not return a valid result.")

    caption = clean_text(results[0]["generated_text"])
    return caption


def build_story_prompt(caption: str, style_instruction: str) -> str:
    """
    Build a stronger prompt for a child-friendly story.
    This version gives stricter length and structure requirements.
    """
    prompt = f"""
You are a children's storyteller.

Write a complete children's story based on the image caption below.

Strict requirements:
- The story MUST be between 60 and 80 words.
- Use very simple English for children aged 3 to 10.
- Make it gentle, fun, and easy to understand.
- Avoid scary, violent, sad, or inappropriate content.
- The story must have a clear beginning, middle, and happy ending.
- Write only the story.
- Do not write a title.
- Do not write bullet points.
- Do not write fewer than 60 words.

Style:
{style_instruction}

Image caption:
{caption}
"""
    return clean_text(prompt)

def generate_story_from_prompt(prompt: str) -> str:
    """
    Generate a story from the prepared prompt.

    This version is more deterministic and better for length control.
    """
    story_pipe = load_story_pipeline()

    try:
        results = story_pipe(
            prompt,
            max_new_tokens=120,
            min_new_tokens=60,
            do_sample=False,
            num_beams=4,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
        )
    except Exception as exc:
        raise RuntimeError("Story generation failed.") from exc

    if not results or "generated_text" not in results[0]:
        raise RuntimeError("Story model returned an invalid result.")

    story = clean_text(results[0]["generated_text"])
    if not story:
        raise RuntimeError("Story model returned an empty story.")

    return story
    
def expand_story_to_target_length(
    short_story: str,
    caption: str,
    style_instruction: str,
) -> str:
    """
    Expand a story that is too short into a full children's story.
    """
    prompt = f"""
You are writing for children aged 3 to 10.

Expand the short story below into a complete children's story.

Requirements:
- Make it between 60 and 80 words.
- Use simple and friendly English.
- Keep it warm, gentle, safe, and easy to understand.
- Keep a happy ending.
- Stay faithful to this image caption: {caption}
- Follow this style: {style_instruction}
- Write only the story.

Short story:
{short_story}
"""
    return generate_story_from_prompt(clean_text(prompt))

def shorten_story_to_target_length(
    long_story: str,
    caption: str,
    style_instruction: str,
) -> str:
    """
    Rewrite a story that is too long into the target length.
    """
    prompt = f"""
You are writing for children aged 3 to 10.

Rewrite the story below into a shorter version.

Requirements:
- Make it between 60 and 80 words.
- Keep the same main idea.
- Use simple and friendly English.
- Keep it safe, warm, and suitable for children.
- Keep a happy ending.
- Stay faithful to this image caption: {caption}
- Follow this style: {style_instruction}
- Write only the story.

Story:
{long_story}
"""
    return generate_story_from_prompt(clean_text(prompt))

def rewrite_story(
    original_story: str,
    caption: str,
    style_instruction: str,
    reason: str,
) -> str:
    """
    Rewrite a story when it is too short, too long, or contains unsafe content.
    """
    repair_prompt = f"""
Rewrite the children's story below.

Reason for rewrite:
{reason}

Requirements:
- Keep the story between 60 and 80 words.
- Use simple English for children aged 3 to 10.
- Make it safe, gentle, fun, and suitable for young children.
- Keep a clear and happy ending.
- Stay aligned with this image caption: {caption}
- Follow this style: {style_instruction}
- Write only the revised story.

Story:
{original_story}
"""
    return generate_story_from_prompt(clean_text(repair_prompt))

def enforce_story_constraints(
    story: str,
    caption: str,
    style_instruction: str,
    max_retries: int = 3,
) -> Tuple[str, Optional[str]]:
    """
    Enforce assignment constraints strictly.

    Rules:
    - Story must be between 50 and 100 words
    - Story must be safe for children
    - If the story still fails after retries, raise an error
      instead of silently returning an invalid result
    """
    warning_message = None
    current_story = clean_text(story)

    for _ in range(max_retries):
        info = get_story_quality_flags(current_story)

        if info["in_range"] and not info["unsafe"]:
            return current_story, warning_message

        if info["unsafe"]:
            current_story = rewrite_story(
                original_story=current_story,
                caption=caption,
                style_instruction=style_instruction,
                reason="The story contains unsafe content for young children.",
            )
            warning_message = "The story was rewritten to make it safer for children."
            continue

        if info["word_count"] < TARGET_MIN_WORDS:
            current_story = expand_story_to_target_length(
                short_story=current_story,
                caption=caption,
                style_instruction=style_instruction,
            )
            warning_message = "The story was expanded automatically to meet the word limit."
            continue

        if info["word_count"] > TARGET_MAX_WORDS:
            current_story = shorten_story_to_target_length(
                long_story=current_story,
                caption=caption,
                style_instruction=style_instruction,
            )
            warning_message = "The story was shortened automatically to meet the word limit."
            continue

    # Final strict validation
    final_info = get_story_quality_flags(current_story)
    if not final_info["in_range"] or final_info["unsafe"]:
        raise ValueError(
            f"Failed to generate a valid story after retries. "
            f"Final word count: {final_info['word_count']}"
        )

    return current_story, warning_message


def text_to_speech_bytes(text: str, lang: str = "en") -> bytes:
    """
    Convert story text to MP3 bytes using gTTS.
    """
    tts = gTTS(text=text, lang=lang)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer.read()


# =========================================================
# 5. UI HELPERS
# =========================================================
def render_header():
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)
    st.write(
        "This app turns a picture into a short story and reads it aloud. "
        "Perfect for young children! ✨"
    )


def render_sidebar():
    st.sidebar.header("⚙️ Story Settings")
    story_style = st.sidebar.selectbox(
        "Choose a story style",
        list(STYLE_OPTIONS.keys())
    )

    show_caption = st.sidebar.checkbox("Show image caption", value=True)
    show_debug = st.sidebar.checkbox("Show debug info", value=False)

    return story_style, show_caption, show_debug


def render_footer():
    st.markdown("---")
    st.caption(
        "Built with Streamlit, Hugging Face Transformers, and gTTS."
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
        type=["png", "jpg", "jpeg", "webp"]
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
                with st.spinner("Creating your magical story..."):
                    start_time = time.time()

                    # Step 1: Caption
                    caption = generate_caption(image)

                    # Step 2: Prompt
                    prompt = build_story_prompt(
                        caption=caption,
                        style_instruction=style_instruction
                    )

                    # Step 3: Story
                    raw_story = generate_story_from_promt(prompt)

                    # Step 4: Constraint enforcement
                    final_story = enforce_story_constraints(
                        story=raw_story,
                        fallback_caption=caption,
                        style_instruction=style_instruction
                    )

                    # Step 5: TTS
                    audio_bytes = text_to_speech_bytes(final_story)

                    elapsed = time.time() - start_time
                    word_count = count_words(final_story)

                st.success("Your story is ready! 🎉")

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
                    mime="text/plain"
                )

                st.download_button(
                    label="📥 Download Audio",
                    data=audio_bytes,
                    file_name="story.mp3",
                    mime="audio/mpeg"
                )

                if show_debug:
                    st.markdown("### 🛠 Debug Info")
                    st.write({
                        "caption_model": CAPTION_MODEL_NAME,
                        "story_model": STORY_MODEL_NAME,
                        "word_count": word_count,
                        "elapsed_seconds": round(elapsed, 2),
                        "style": story_style_label,
                    })

                if word_count < TARGET_MIN_WORDS or word_count > TARGET_MAX_WORDS:
                    st.warning(
                        f"Story word count is {word_count}, which is outside the preferred range "
                        f"({TARGET_MIN_WORDS}-{TARGET_MAX_WORDS})."
                    )

        except Exception as e:
            st.error("Something went wrong while generating the story.")
            st.exception(e)

    else:
        st.info("Upload a picture to begin your story adventure!")

    render_footer()


# =========================================================
# 7. ENTRY POINT
# =========================================================
if __name__ == "__main__":
    main()
