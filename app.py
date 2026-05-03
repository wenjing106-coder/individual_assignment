"""
Magic Story Maker
=================
A Streamlit web application that transforms any uploaded image into a
short, child-friendly story and reads it aloud.

Pipeline (three sequential steps):
    1. Image Captioning  – microsoft/git-base-coco describes the scene.
    2. Story Generation  – Qwen/Qwen2.5-0.5B-Instruct writes a 60-90 word
                           story for children aged 4-8, guided by the chosen
                           style (Warm & Happy / Adventure / Bedtime).
    3. Text-to-Speech    – gTTS converts the final story to an MP3 audio clip.

Memory management strategy:
    Each model is loaded, used, then immediately deleted and garbage-collected
    so that at most one model occupies RAM at a time.  Peak usage is ~728 MB
    (caption model), well within Streamlit Cloud's ~1 GB free-tier limit.

Module layout:
    Section 1 – Page configuration and application-level constants.
    Section 2 – Pure utility functions (text helpers, image validation).
    Section 3 – Model execution functions (caption → story → TTS).
    Section 4 – Post-processing / constraint enforcement.
    Section 5 – Streamlit UI helpers (header, sidebar, story card, footer).
    Section 6 – Main application orchestration function.
    Section 7 – Entry point guard.

Author : Magic Story Maker Project
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------
import gc          # Manual garbage collection to free model memory promptly
import io          # In-memory byte stream used for TTS audio buffer
import re          # Regular expressions for word counting and text cleaning
import time        # Wall-clock timing for performance debug display

# ---------------------------------------------------------------------------
# Type-hint helpers
# ---------------------------------------------------------------------------
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import streamlit as st                                    # Web UI framework
import torch                                              # PyTorch tensor ops
from gtts import gTTS                                     # Google TTS (HTTPS)
from PIL import Image, UnidentifiedImageError             # Image I/O via Pillow
from transformers import (                                # Hugging Face models
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)


# ===========================================================================
# SECTION 1 – PAGE CONFIGURATION AND APPLICATION CONSTANTS
# ===========================================================================

# Must be the first Streamlit call in the script.
st.set_page_config(
    page_title="Magic Story Maker",
    page_icon="🌈",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Word-count targets
# TARGET_MIN_WORDS  : stories below this threshold trigger an informational
#                     warning (story is still shown – never silently hidden).
# TARGET_MAX_WORDS  : soft upper bound displayed in the word-count warning.
# TARGET_HARD_MAX   : hard ceiling enforced by sentence-boundary truncation;
#                     set slightly below TARGET_MAX_WORDS to leave a margin.
# ---------------------------------------------------------------------------
TARGET_MIN_WORDS: int = 50
TARGET_MAX_WORDS: int = 110
TARGET_HARD_MAX: int  = 100

# ---------------------------------------------------------------------------
# Model identifiers
#
# CAPTION_MODEL_NAME
#   microsoft/git-base-coco  (~182 M parameters, ~728 MB fp32 RAM)
#   GIT (Generative Image-to-Text) decoder conditioned on CLIP visual tokens,
#   fine-tuned on the COCO dataset.  Produces scene-aware captions such as
#   "a woman sitting in a golden carriage surrounded by swans".
#   Loaded first; freed before the story model is instantiated.
#
# STORY_MODEL_NAME
#   Qwen/Qwen2.5-0.5B-Instruct  (~494 M parameters, ~500 MB fp16 RAM)
#   Decoder-only causal LM with instruction fine-tuning.  Unlike seq2seq
#   models (T5, LaMini-Flan-T5), a decoder LM appends tokens to the
#   assistant turn rather than transforming its input, so prompt rules
#   never appear verbatim in the generated story.  The built-in chat
#   template cleanly separates system / user / assistant roles.
#   Loaded after the caption model has been freed; peak RAM ≈ 728 MB ✅.
# ---------------------------------------------------------------------------
CAPTION_MODEL_NAME: str = "microsoft/git-base-coco"
STORY_MODEL_NAME: str   = "Qwen/Qwen2.5-0.5B-Instruct"

# ---------------------------------------------------------------------------
# Text-to-Speech settings (gTTS – zero local RAM, uses outbound HTTPS)
# ---------------------------------------------------------------------------
TTS_LANGUAGE: str     = "en"   # BCP-47 language tag passed to gTTS
TTS_SLOW_SPEECH: bool = False  # False = normal reading speed for children

# ---------------------------------------------------------------------------
# Child-safety: words that must not appear in the final story.
# Uses whole-word regex matching to avoid false positives on substrings
# (e.g. the word "dead" inside "instead").
# ---------------------------------------------------------------------------
BANNED_TERMS: List[str] = [
    "blood", "kill", "dead", "gun", "knife",
    "monster", "horror", "terror",
    "violent", "violence", "war", "hate",
]

# ---------------------------------------------------------------------------
# Story style options
# Each key is the human-readable label shown in the sidebar.
# Each value is a 2-tuple: (tone_description, ending_instruction).
# Both strings are injected into the LLM system message.
# ---------------------------------------------------------------------------
STYLE_OPTIONS: Dict[str, Tuple[str, str]] = {
    "Warm & Happy 😊": (
        "warm and cheerful",
        "The story ends with everyone smiling and feeling happy.",
    ),
    "Adventure 🚀": (
        "exciting and playful",
        "The story ends with a safe, joyful adventure completed.",
    ),
    "Bedtime 🌙": (
        "calm and soothing",
        "The story ends peacefully, like a gentle lullaby sending "
        "the characters to sleep.",
    ),
}

# ---------------------------------------------------------------------------
# CSS gradient backgrounds – one per style.
# Used in both the header banner (U1) and the story card (U2) so the
# colour theme changes automatically when the user picks a different style.
# ---------------------------------------------------------------------------
STYLE_GRADIENTS: Dict[str, str] = {
    "Warm & Happy 😊": "linear-gradient(135deg, #D4B3F5 0%, #8EC5FC 100% )",
    "Adventure 🚀":    "linear-gradient(135deg, #FFECD2 0%, #FCB69F 100% )",
    "Bedtime 🌙":      "linear-gradient(135deg, #C2E9FB 100%,#A1C4FD 20% )",
}

# Fallback gradient used when a style key is not found in STYLE_GRADIENTS.
DEFAULT_GRADIENT: str = STYLE_GRADIENTS["Warm & Happy 😊"]


# ===========================================================================
# SECTION 2 – UTILITY FUNCTIONS
# ===========================================================================

def count_words(text: str) -> int:
    """Return the number of word tokens in *text*.

    Uses a regex that matches contractions (e.g. "it's") and hyphenated
    compounds as single tokens, consistent with how the LLM counts words.

    Args:
        text: Any string to be counted.

    Returns:
        Integer word count (0 for an empty or whitespace-only string).
    """
    return len(re.findall(r"\b[\w'-]+\b", text))


def contains_unsafe_content(text: str) -> bool:
    """Check whether *text* contains any term from BANNED_TERMS.

    Performs whole-word matching (\\b anchors) to avoid false positives on
    substrings – for example, "dead" would match in "dead" but NOT in the
    word "instead".

    Args:
        text: The string to scan (typically the generated story).

    Returns:
        True if at least one banned term is found; False otherwise.
    """
    lowered_text = text.lower()
    return any(
        re.search(r"\b" + re.escape(term) + r"\b", lowered_text)
        for term in BANNED_TERMS
    )


def clean_text(text: str) -> str:
    """Normalise whitespace in *text*.

    Collapses any sequence of whitespace characters (spaces, tabs, newlines)
    into a single space and strips leading / trailing whitespace.  Called
    after every model output to produce a clean, single-line string.

    Args:
        text: Raw model output or any string to normalise.

    Returns:
        Whitespace-normalised string.
    """
    return re.sub(r"\s+", " ", text).strip()


def validate_image(uploaded_file) -> Tuple[bool, Optional[str]]:
    """Validate that *uploaded_file* is a supported image type.

    Checks the MIME type reported by Streamlit's UploadedFile object.
    Does NOT open or decode the image – that is handled separately by
    ``safe_open_image`` to provide more granular error messages.

    Args:
        uploaded_file: A ``streamlit.runtime.uploaded_file_manager.UploadedFile``
                       object, or None if no file has been uploaded.

    Returns:
        A 2-tuple ``(is_valid, error_message)`` where:
        - ``is_valid`` is True when the file passes validation.
        - ``error_message`` is a human-readable string when validation fails,
          or None when it passes.
    """
    if uploaded_file is None:
        return False, "Please upload an image first."

    supported_mime_types = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
    if uploaded_file.type not in supported_mime_types:
        return False, "Only PNG, JPG, JPEG, and WEBP images are supported."

    return True, None


def safe_open_image(uploaded_file) -> Image.Image:
    """Open *uploaded_file* with Pillow and convert it to RGB colour mode.

    Wraps Pillow's ``Image.open`` with specific exception handling so that
    callers receive a clear ``ValueError`` with a user-friendly message
    rather than a raw Pillow exception.

    Args:
        uploaded_file: A Streamlit UploadedFile object that has already
                       passed ``validate_image``.

    Returns:
        A ``PIL.Image.Image`` object in RGB mode.

    Raises:
        ValueError: If the file cannot be decoded as a valid image.
    """
    try:
        return Image.open(uploaded_file).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc
    except Exception as exc:
        raise ValueError("Failed to read the uploaded image.") from exc


def truncate_at_sentence_boundary(text: str, max_words: int) -> str:
    """Shorten *text* to at most *max_words* words, preserving sentence ends.

    Iterates through sentences (split on ``.``, ``!``, or ``?`` followed by
    whitespace) and keeps complete sentences while the running word count
    stays within *max_words*.  If not even a single sentence fits, falls
    back to a word-level cut with a closing period appended.

    Args:
        text:      The story string to truncate.
        max_words: Maximum number of words allowed in the output.

    Returns:
        Truncated string that ends at a sentence boundary (or a word
        boundary with a period appended as a last resort).
    """
    # Split on sentence-ending punctuation followed by whitespace.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    kept_sentences: List[str] = []
    running_word_count = 0

    for sentence in sentences:
        sentence_word_count = count_words(sentence)
        if running_word_count + sentence_word_count <= max_words:
            kept_sentences.append(sentence)
            running_word_count += sentence_word_count
        else:
            break  # Adding this sentence would exceed the limit

    if kept_sentences:
        return " ".join(kept_sentences)

    # Fallback: no complete sentence fits – cut at word boundary.
    words = text.split()
    truncated = " ".join(words[:max_words]).rstrip(",;:-")
    # Ensure the fallback result ends with sentence-closing punctuation.
    if not truncated.endswith((".", "!", "?")):
        truncated += "."
    return truncated


# ===========================================================================
# SECTION 3 – MODEL EXECUTION FUNCTIONS  (load → run → free)
# ===========================================================================
#
# Memory budget on Streamlit Cloud free tier (~1 GB usable RAM)
# ──────────────────────────────────────────────────────────────
#   Step 1  git-base-coco      (caption)  ~728 MB fp32  → del + gc.collect()
#   Step 2  Qwen2.5-0.5B-Inst  (story)    ~500 MB fp16  → del + gc.collect()
#   Step 3  gTTS               (audio)      ~0 MB HTTPS
#
#   Peak RAM = max(728 MB, 500 MB) = 728 MB  ✅  (well within 1 GB)
#
# Each function follows the same pattern:
#   1. Load the model / pipeline.
#   2. Run inference inside a try block.
#   3. Delete the model and call gc.collect() in the finally block so that
#      memory is released even if an exception occurs mid-inference.

def run_caption_model(image: Image.Image) -> str:
    """Generate a scene description for *image* using GIT-base-COCO.

    Loads the microsoft/git-base-coco image-to-text pipeline, runs a
    single forward pass, then frees all model weights.  Peak RAM during
    this step is approximately 728 MB.

    Args:
        image: A PIL RGB image to caption.

    Returns:
        A cleaned, single-line caption string (e.g. "a woman sitting in
        a golden carriage surrounded by swans").

    Raises:
        ValueError: If the model returns an empty or malformed result.
    """
    caption_pipeline = pipeline(
        task="image-to-text",
        model=CAPTION_MODEL_NAME,
    )
    try:
        model_output = caption_pipeline(image)

        # Guard against unexpected output format from the pipeline.
        if not model_output or "generated_text" not in model_output[0]:
            raise ValueError("Caption model returned an empty or invalid result.")

        raw_caption = model_output[0]["generated_text"]
        return clean_text(raw_caption)

    finally:
        # Always free the pipeline regardless of success or failure.
        del caption_pipeline
        gc.collect()


def build_chat_messages(
    caption: str,
    style_tone: str,
    style_ending: str,
) -> List[Dict[str, str]]:
    """Construct the chat-template message list for Qwen2.5-0.5B-Instruct.

    Separates storytelling constraints into the system message and the
    image scene into the user message.  This split is critical: Qwen's
    decoder treats the system message as background context it should
    *follow*, not reproduce.  Earlier seq2seq models (LaMini-Flan-T5)
    copied rule text verbatim because they could not distinguish
    instructions from story content.

    Design decisions:
    - No full few-shot examples: prevents the model from copying example
      sentences instead of generating original content.
    - Style tone and ending are injected as natural sentences, not
      numbered rules, to avoid the model echoing the rule numbers.
    - Length constraint (60-90 words) is stated once in the system message.

    Args:
        caption:      The image caption produced by the captioning step.
        style_tone:   Short description of the desired narrative tone
                      (e.g. "warm and cheerful").
        style_ending: One-sentence instruction describing how the story
                      should conclude (e.g. "The story ends with everyone
                      smiling and feeling happy.").

    Returns:
        A list of message dicts in OpenAI / Qwen chat format:
        ``[{"role": "system", "content": ...},
           {"role": "user",   "content": ...}]``
    """
    system_message = (
        "You are a kind and imaginative children's storyteller. "
        "When given a scene description, you write a short, original story "
        "for children aged 4 to 8. "
        "Always use simple, everyday words a young child understands. "
        f"Your stories are {style_tone}. "
        f"{style_ending} "
        "Write between 60 and 90 words. "
        "Do not include a title. Do not repeat sentences."
    )
    user_message = (
        "Write a children's story about this scene:\n"
        + caption
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user",   "content": user_message},
    ]


def run_story_model(caption: str, style_label: str) -> str:
    """Generate a children's story from *caption* using Qwen2.5-0.5B-Instruct.

    Loads the story model in fp16 on CPU (~500 MB RAM), applies the
    built-in Qwen chat template, runs greedy-sampled generation, then
    frees all weights before returning.

    Why Qwen2.5-0.5B-Instruct outperforms LaMini-Flan-T5-248M here:
    - Decoder-only architecture: the model only generates tokens that
      follow the assistant turn marker, so prompt rules are never echoed.
    - Instruction fine-tuning: the model understands "write a children's
      story" and respects length / style constraints without repeating them.
    - fp16 loading keeps peak RAM at ~500 MB, safely within the 728 MB
      already occupied (and then freed) by the caption step.

    Generation parameters:
    - do_sample=True / temperature=0.8 : controlled creativity without
      wild hallucinations; pure beam search on a small model tends to
      produce repetitive high-probability phrases.
    - top_p=0.9                        : nucleus sampling filters the very
      long tail of low-probability tokens.
    - repetition_penalty=1.15          : mild penalty prevents successive
      sentences from opening with the same phrase.
    - max_new_tokens=180               : generous ceiling; post-processing
      trims if needed.
    - min_new_tokens=60                : prevents a one-sentence output.

    Args:
        caption:     The image caption from ``run_caption_model``.
        style_label: Key into STYLE_OPTIONS (e.g. "Warm & Happy 😊").

    Returns:
        A cleaned story string ready for constraint enforcement.
    """
    style_tone, style_ending = STYLE_OPTIONS[style_label]
    chat_messages = build_chat_messages(caption, style_tone, style_ending)

    # Load tokeniser and model separately so we can delete both explicitly.
    tokenizer = AutoTokenizer.from_pretrained(STORY_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        STORY_MODEL_NAME,
        torch_dtype=torch.float16,  # fp16 halves memory vs fp32
        device_map="cpu",           # CPU inference; no GPU required
    )

    try:
        # Apply Qwen's chat template to produce a formatted prompt string.
        formatted_prompt = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,  # Appends the assistant turn marker
        )
        tokenised_input = tokenizer(formatted_prompt, return_tensors="pt")
        prompt_length = tokenised_input["input_ids"].shape[1]

        with torch.no_grad():
            generated_ids = model.generate(
                **tokenised_input,
                max_new_tokens=180,
                min_new_tokens=60,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens – slice off the prompt so
        # the system / user messages never appear in the story output.
        new_token_ids = generated_ids[0][prompt_length:]
        raw_story = tokenizer.decode(new_token_ids, skip_special_tokens=True)
        return clean_text(raw_story)

    finally:
        del model, tokenizer
        gc.collect()


def run_tts(story_text: str) -> bytes:
    """Convert *story_text* to MP3 audio bytes via Google TTS (gTTS).

    Uses the gTTS library, which makes an outbound HTTPS request to
    Google's TTS API and streams the MP3 back.  No local model weights
    are downloaded; RAM impact is negligible.  Streamlit Cloud allows
    outbound HTTPS, so this works without any special configuration.

    Args:
        story_text: The final, constraint-checked story string.

    Returns:
        Raw MP3 audio as a ``bytes`` object, suitable for
        ``st.audio(audio_bytes, format="audio/mp3")``.
    """
    tts_engine = gTTS(text=story_text, lang=TTS_LANGUAGE, slow=TTS_SLOW_SPEECH)
    audio_buffer = io.BytesIO()
    tts_engine.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer.read()


# ===========================================================================
# SECTION 4 – POST-PROCESSING AND CONSTRAINT ENFORCEMENT
# ===========================================================================

def enforce_story_constraints(raw_story: str) -> Tuple[str, Optional[str]]:
    """Apply safety, length, and quality constraints to *raw_story*.

    Three checks are applied in priority order:

    1. **Safety check** – if the story contains any term from BANNED_TERMS,
       it is replaced entirely with a guaranteed child-safe fallback sentence.
       This is a hard block; no further checks are run on the fallback.

    2. **Hard-length ceiling** – if the word count exceeds TARGET_HARD_MAX,
       the story is trimmed to the nearest sentence boundary so that it
       never reads as abruptly cut mid-sentence.

    3. **Soft minimum** – if the word count is below TARGET_MIN_WORDS, an
       informational warning is returned but the short story is still shown
       (it may still be enjoyable; hiding it would be worse UX).

    No additional model calls are made in this function.

    Args:
        raw_story: The story string returned by ``run_story_model``.

    Returns:
        A 2-tuple ``(final_story, warning_message)`` where:
        - ``final_story`` is the processed story (safe and within limits).
        - ``warning_message`` is a user-facing string if a constraint was
          triggered, or None if the story passed all checks cleanly.
    """
    processed_story = clean_text(raw_story)
    warning_message: Optional[str] = None

    # --- Check 1: Child-safety filter ---
    if contains_unsafe_content(processed_story):
        safe_fallback = (
            "Once upon a time, a little friend went on a gentle adventure "
            "and came home happy, warm, and full of joy."
        )
        return safe_fallback, "The story was replaced with a child-safe version."

    current_word_count = count_words(processed_story)

    # --- Check 2: Hard length ceiling ---
    if current_word_count > TARGET_HARD_MAX:
        processed_story = truncate_at_sentence_boundary(
            processed_story, TARGET_HARD_MAX
        )
        warning_message = "The story was lightly trimmed to keep it short and sweet."

    # --- Check 3: Soft minimum warning ---
    elif current_word_count < TARGET_MIN_WORDS:
        warning_message = (
            f"The story is a little short ({current_word_count} words) "
            "but should still be enjoyable!"
        )

    return processed_story, warning_message


# ===========================================================================
# SECTION 5 – STREAMLIT UI HELPER FUNCTIONS
# ===========================================================================

def render_header(style_label: str = "Warm & Happy 😊") -> None:
    """Render the gradient banner at the top of the main page (U1).

    Injects a CSS-styled ``<div>`` with a gradient background that matches
    the currently selected story style.  Using ``unsafe_allow_html=True``
    is safe here because the only dynamic value injected is a CSS gradient
    string sourced from the internal STYLE_GRADIENTS constant, not from
    user input.

    Args:
        style_label: The currently selected style key from STYLE_OPTIONS.
                     Controls which gradient is applied to the banner.
    """
    banner_gradient = STYLE_GRADIENTS.get(style_label, DEFAULT_GRADIENT)
    st.markdown(
        f"""
        <div style="
            background: {banner_gradient};
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


def render_story_card(story_text: str, style_label: str) -> None:
    """Display *story_text* inside a styled gradient card (U2).

    The card uses the same gradient as the header banner, creating a
    coherent colour theme across the page.  A coloured left border provides
    a visual anchor that draws the reader's eye to the story text.

    Args:
        story_text:  The final, processed story string to display.
        style_label: The currently selected style key, used to look up
                     the matching gradient from STYLE_GRADIENTS.
    """
    card_gradient = STYLE_GRADIENTS.get(style_label, DEFAULT_GRADIENT)
    st.markdown(
        f"""
        <div style="
            background: {card_gradient};
            border-radius: 14px;
            border-left: 6px solid rgba(0,0,0,0.12);
            padding: 1.4rem 1.8rem;
            margin: 0.5rem 0 1.2rem 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.07);
            font-size: 1.08rem;
            line-height: 1.75;
            color: #2d2d2d;
        ">
            {story_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> Tuple[str, bool, bool]:
    """Render the sidebar with style selection and advanced options (U4).

    Layout:
    - A decorative gradient banner heading.
    - A radio button group for story style (replaces the previous selectbox
      so all three options are visible at a glance without a dropdown).
    - A collapsed expander for less-frequently used options (caption display
      and debug info), keeping the sidebar uncluttered by default.

    Widget keys (``key=`` parameter) are required for checkboxes inside
    ``st.expander`` to ensure Streamlit preserves their state correctly
    across reruns triggered by other widgets.

    Returns:
        A 3-tuple ``(story_style_label, show_caption, show_debug)`` where:
        - ``story_style_label`` is the selected key from STYLE_OPTIONS.
        - ``show_caption``      is True if the image caption should be shown.
        - ``show_debug``        is True if the debug info panel should be shown.
    """
    # Decorative sidebar header banner
    st.sidebar.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #FFECD2, #FCB69F);
            border-radius: 12px;
            padding: 0.8rem 1rem;
            margin-bottom: 1rem;
            text-align: center;
        ">
            <span style="font-size: 1.4rem;">⚙️</span>
            <span style="font-weight: 700; font-size: 1rem; color: #2d2d2d;">
              &nbsp;Story Settings
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Radio group: one visible option per style (U4)
    selected_style = st.sidebar.radio(
        "🎨 Story Style",
        options=list(STYLE_OPTIONS.keys()),
        horizontal=False,  # Vertical layout suits the narrow sidebar width
        index=0,
    )

    st.sidebar.markdown("---")

    # Collapsed expander hides rarely-needed options from the default view
    with st.sidebar.expander("🔧 Advanced Options", expanded=False):
        show_image_caption = st.checkbox(
            "Show image caption",
            value=True,
            key="show_caption_cb",  # Stable key preserves state across reruns
        )
        show_debug_panel = st.checkbox(
            "Show debug info",
            value=False,
            key="show_debug_cb",
        )

    return selected_style, show_image_caption, show_debug_panel


def render_upload_area() -> None:
    """Render a decorative dashed-border hint above the file uploader (U6).

    Uses an HTML ``<div>`` with a dashed border and centred text to create
    a recognisable drop-zone visual that guides users (including children's
    parents) to upload their picture.  The actual Streamlit file uploader
    widget follows immediately beneath this hint box.
    """
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
                <br>
                <span style="color: #aaa; font-size: 0.85rem;">
                    PNG · JPG · WEBP supported
                </span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Render a centred footer crediting the libraries and models used."""
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #888; font-size: 0.82rem;'>"
        "Built with using "
        "<b>Streamlit</b> · <b>GIT-base-COCO</b> · "
        "<b>Qwen2.5-0.5B-Instruct</b> · <b>gTTS</b>"
        "</p>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# SECTION 6 – MAIN APPLICATION ORCHESTRATION
# ===========================================================================

def main() -> None:
    """Orchestrate the full Magic Story Maker workflow.

    Execution flow:
        1. Render sidebar (must precede header so the selected style is
           available for the gradient banner colour).
        2. Render gradient banner header.
        3. Render upload area hint + file uploader widget.
        4. Validate the uploaded image; display it.
        5. Wait for the user to press "Create My Story".
        6. Run the three-step pipeline with a live progress bar:
               Step 1/3 – Image captioning  (run_caption_model)
               Step 2/3 – Story generation  (run_story_model)
               Step 3/3 – Text-to-speech    (run_tts)
        7. Post-process the story (enforce_story_constraints).
        8. Display the story card, audio player, and download buttons.
        9. Optionally show debug information.
       10. Render footer.

    Early-return guards are used after each validation step so that the
    main pipeline code is not indented inside nested ``if`` blocks.
    """

    # --- Sidebar and header (sidebar first to capture style for gradient) ---
    selected_style_label, show_image_caption, show_debug_panel = render_sidebar()
    render_header(selected_style_label)

    # --- Upload area (U6) ---
    render_upload_area()

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",  # The hint div above acts as the label
    )

    # Guard: no file yet – show a friendly prompt and stop.
    if uploaded_file is None:
        st.info("📸 Upload a picture above to begin your story adventure! 🌟")
        render_footer()
        return

    # Guard: unsupported file type.
    image_is_valid, validation_error = validate_image(uploaded_file)
    if not image_is_valid:
        st.error(validation_error)
        render_footer()
        return

    # Guard: file is declared as an image type but cannot be decoded.
    try:
        pil_image = safe_open_image(uploaded_file)
    except ValueError as decode_error:
        st.error(str(decode_error))
        render_footer()
        return

    st.image(pil_image, caption="Your uploaded image", use_container_width=True)

    # Primary action button (U6: primary style + full width)
    if not st.button("✨ Create My Story", use_container_width=True, type="primary"):
        render_footer()
        return

    # -------------------------------------------------------------------------
    # Three-step pipeline with live progress indicator (U3)
    # -------------------------------------------------------------------------
    progress_bar = st.progress(0, text="Starting…")
    step_status  = st.empty()   # Reused placeholder for the current step label

    # ------------------------------------------------------------------
    # Step 1 / 3 – Image Captioning
    # ------------------------------------------------------------------
    step_status.markdown("**Step 1 / 3** &nbsp;🔍&nbsp; Reading the picture…",
                         unsafe_allow_html=True)
    progress_bar.progress(5, text="Step 1 / 3 — Reading the picture…")

    caption_start_time = time.time()
    image_caption      = run_caption_model(pil_image)
    caption_duration   = time.time() - caption_start_time

    progress_bar.progress(35, text="Step 1 / 3 — Done ✅")

    # Optionally reveal the raw caption for the user's reference.
    if show_image_caption:
        with st.expander("🖼️ Image Caption", expanded=True):
            st.write(image_caption)

    # ------------------------------------------------------------------
    # Step 2 / 3 – Story Generation
    # ------------------------------------------------------------------
    step_status.markdown("**Step 2 / 3** &nbsp;📝&nbsp; Writing your story…",
                         unsafe_allow_html=True)
    progress_bar.progress(40, text="Step 2 / 3 — Writing your story…")

    story_start_time = time.time()
    raw_story        = run_story_model(image_caption, selected_style_label)
    story_duration   = time.time() - story_start_time

    # Apply safety, length, and quality constraints (no extra model calls).
    final_story, constraint_warning = enforce_story_constraints(raw_story)

    progress_bar.progress(75, text="Step 2 / 3 — Done ✅")

    # ------------------------------------------------------------------
    # Step 3 / 3 – Text-to-Speech
    # ------------------------------------------------------------------
    step_status.markdown("**Step 3 / 3** &nbsp;🔊&nbsp; Recording the story…",
                         unsafe_allow_html=True)
    progress_bar.progress(80, text="Step 3 / 3 — Recording the story…")

    tts_start_time = time.time()
    audio_bytes    = run_tts(final_story)
    tts_duration   = time.time() - tts_start_time

    progress_bar.progress(100, text="All done! 🎉")
    step_status.empty()   # Clear the step label once all steps are complete

    # -------------------------------------------------------------------------
    # Results display
    # -------------------------------------------------------------------------
    total_elapsed_time = caption_duration + story_duration + tts_duration
    final_word_count   = count_words(final_story)

    st.success(
        f"Your story is ready! 🎉  "
        f"({final_word_count} words · {total_elapsed_time:.0f} s)"
    )

    # Show a user-friendly notice if a constraint was triggered.
    if constraint_warning:
        st.info(constraint_warning)

    # Story card with gradient background (U2)
    st.markdown("### 📖 Your Story")
    render_story_card(final_story, selected_style_label)

    # Audio player + download buttons in a single three-column row (U5)
    st.markdown("### 🔊 Listen & Download")
    col_audio, col_download_text, col_download_audio = st.columns([3, 1, 1])

    with col_audio:
        st.audio(audio_bytes, format="audio/mp3")

    with col_download_text:
        st.download_button(
            label="📄 Download Your Story",
            data=final_story,
            file_name="story.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with col_download_audio:
        st.download_button(
            label="🎵 Download Your Audio",
            data=audio_bytes,
            file_name="story.mp3",
            mime="audio/mpeg",
            use_container_width=True,
        )

    # Optional debug information panel
    if show_debug_panel:
        with st.expander("🛠 Debug Info", expanded=False):
            st.write({
                "caption_model":  CAPTION_MODEL_NAME,
                "story_model":    STORY_MODEL_NAME,
                "story_arch":     "Qwen chat-template, single forward pass",
                "raw_caption":    image_caption,
                "word_count":     final_word_count,
                "t_caption_s":    round(caption_duration, 1),
                "t_story_s":      round(story_duration, 1),
                "t_tts_s":        round(tts_duration, 1),
                "total_s":        round(total_elapsed_time, 1),
                "style":          selected_style_label,
            })

    # Word-count range warning (separate from the constraint warning above)
    if final_word_count < TARGET_MIN_WORDS or final_word_count > TARGET_MAX_WORDS:
        st.warning(
            f"Story word count is {final_word_count} "
            f"(target range: {TARGET_MIN_WORDS}–{TARGET_MAX_WORDS} words)."
        )

    render_footer()


# ===========================================================================
# SECTION 7 – ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    main()
