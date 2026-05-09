"""
ISOM5240 Individual Assignment
Name: WEN Jing
SID: 21121986

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


Module layout:
    Section 1 – Page configuration and application-level constants.
    Section 2 – Pure utility functions (text helpers, image validation).
    Section 3 – Model execution functions (caption → story → TTS).
    Section 4 – Post-processing / constraint enforcement.
    Section 5 – Streamlit UI helpers (header, sidebar, story card, footer).
    Section 6 – Main application orchestration function.
    Section 7 – Entry point guard.

"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------
import gc          # Manual garbage collection to free model memory promptly
import io          # In-memory byte stream used for TTS audio buffer
import re          # Regular expressions for word counting and text cleaning
import time        # Wall-clock timing for performance debug display

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

st.set_page_config(
    page_title="Magic Story Maker",
    page_icon="🌈",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Word-count targets 50-110 words
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
# ---------------------------------------------------------------------------
BANNED_TERMS: List[str] = [
    "blood", "kill", "dead", "gun", "knife",
    "monster", "horror", "terror",
    "violent", "violence", "war", "hate",
]

# ---------------------------------------------------------------------------
# Story style options – each key is a user-facing label shown in the sidebar
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
# ---------------------------------------------------------------------------
STYLE_GRADIENTS: Dict[str, str] = {
    "Warm & Happy 😊": "linear-gradient(135deg, #D4B3F5 0%, #8EC5FC 100% )",
    "Adventure 🚀":    "linear-gradient(135deg, #FFECD2 0%, #FCB69F 100% )",
    "Bedtime 🌙":      "linear-gradient(135deg, #A1C4FD 20%, #C2E9FB 100% )",
}

# Fallback gradient used when a style key is not found in STYLE_GRADIENTS.
DEFAULT_GRADIENT: str = STYLE_GRADIENTS["Warm & Happy 😊"]


# ===========================================================================
# SECTION 2 – UTILITY FUNCTIONS
# ===========================================================================

def count_words(text: str) -> int:
    """Return the number of word tokens in *text*.
    Returns:
        Integer word count (0 for an empty or whitespace-only string).
    """
    return len(re.findall(r"\b[\w'-]+\b", text))


def contains_unsafe_content(text: str) -> bool:
    """Check whether *text* contains any term from BANNED_TERMS.
    Returns:
        True if at least one banned term is found; False otherwise.
    """
    lowered_text = text.lower()
    return any(
        re.search(r"\b" + re.escape(term) + r"\b", lowered_text)
        for term in BANNED_TERMS
    )


def clean_text(text: str) -> str:
    """Normalise whitespace (spaces, tabs, newlines) in *text*.
    Returns:
        Whitespace-normalised string.
    """
    return re.sub(r"\s+", " ", text).strip()


def validate_image(uploaded_file) -> Tuple[bool, Optional[str]]:
    """Validate that *uploaded_file* is a supported image type.
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
# SECTION 3 – MODEL EXECUTION FUNCTIONS
# ===========================================================================
#
# Each function follows the same pattern:
#   1. Load the model / pipeline.
#   2. Run inference inside a try block.
#   3. Delete the model and call gc.collect() in the finally block so that
#      memory is released even if an exception occurs mid-inference, optimizing for streamlit RAM limitations.

def run_caption_model(image: Image.Image) -> str:
    """Generate a scene description for *image* using GIT-base-COCO.
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
    """Apply safety, length, and quality constraints in order to *raw_story*.

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
    """Render the gradient banner at the top of the main page.
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
                color: #3A3A3A;
            ">Select story style in the sidebar & Upload a picture and create a fun story for kids! ✨</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_story_card(story_text: str, style_label: str) -> None:
    """Display *story_text* inside a styled gradient card.
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
    """Render the sidebar with style selection and advanced options.
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
            <span style="font-weight: 700; font-size: 1.4rem; color: #2d2d2d;">
              &nbsp;Story Settings
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Radio group: one visible option per style
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
    """
    st.markdown(
        """
        <div style="
            background: #f9f9f9;
            border: 2px #d0d0d0;
            border-radius: 14px;
            padding: 1.2rem 1.5rem 0.8rem 1.5rem;
            margin-bottom: 0.5rem;
            text-align: center;
        ">
            <div style="font-size: 2rem;">🖼️</div>
            <p style="margin: 0.3rem 0 0 0; color: #666; font-size: 0.95rem;">
                Drag &amp; drop your picture below or click the button below
                <br>
                <span style="color: #aaa; font-size: 0.85rem;">
                    PNG · JPG ·JPEG · WEBP supported
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

    # Guard: no file yet – show style-reminder banner and friendly prompt, then stop.
    if uploaded_file is None:
        # The banner is shown only at this stage so it does not clutter the
        # results view after a story has been generated.
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

    # Action button styled with a muted sage-green (#7A9E87) instead of
    # Streamlit's default high-saturation red/orange primary colour.
    # We target the button by a stable data-testid + key attribute pair:
    # Streamlit sets `data-testid="baseButton-secondary"` on default buttons
    # and inserts the `key` value as the element's `id`, so the selector
    # `#create_story_btn` is both unique and version-stable.
    st.markdown(
        """
        <style>
        /* Muted sage-green for the Create My Story button only.
           The key="create_story_btn" causes Streamlit to render the button
           inside a wrapper whose immediate <button> child carries that key
           in its aria-label, making this selector precise and safe. */
        div[data-testid="stButton"]:has(> button[data-testid="baseButton-secondary"])
            > button[aria-label="✨ Create My Story"],
        div[data-testid="stButton"]
            > button[data-testid="baseButton-secondary"] {
            background-color: #7A9E87 !important;
            color: #ffffff !important;
            border: 1px solid #6A8E77 !important;
        }
        div[data-testid="stButton"]
            > button[data-testid="baseButton-secondary"]:hover {
            background-color: #6A8E77 !important;
            border-color: #5A7E67 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if not st.button("✨ Create My Story", use_container_width=True,
                     key="create_story_btn"):
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
            label="📄 Save Story",
            data=final_story,
            file_name="story.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with col_download_audio:
        st.download_button(
            label="🎵 Save Audio",
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
