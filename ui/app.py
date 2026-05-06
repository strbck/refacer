"""
ui/app.py
~~~~~~~~~
Local Gradio interface for Refacer.

Run from the repo root:
    python ui/app.py

Opens a browser tab at http://127.0.0.1:7860
All processing is fully offline — no network calls are made at runtime.
"""

import logging
import os
import shutil
import sys

# ui/ is one level below the repo root; the package lives at the repo root,
# so its parent must be on sys.path for `import refacer` to resolve.
_HERE = os.path.dirname(os.path.abspath(__file__))       # .../refacer/ui
_PACKAGE_DIR = os.path.dirname(_HERE)                    # .../refacer
_PARENT_DIR = os.path.dirname(_PACKAGE_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

import gradio as gr

from refacer import metadata
from refacer.models import load_models
from refacer.pipeline import SUPPORTED_EXTENSIONS, RunStats, run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(_PACKAGE_DIR, "models")
INPUT_DIR = os.path.join(_PACKAGE_DIR, "input")
OUTPUT_DIR = os.path.join(_PACKAGE_DIR, "output")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Model loading — once at startup
# ---------------------------------------------------------------------------
print("\nLoading models — this may take a moment on first run…")
try:
    MODELS = load_models(MODELS_DIR)
    print("Models loaded. Starting UI…\n")
except (FileNotFoundError, ImportError) as exc:
    print(f"\nERROR: {exc}\n", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_stats(stats: RunStats) -> str:
    """Turn a RunStats into a readable log string for the UI."""
    lines = [str(stats)]
    for img_result in stats.image_results:
        lines.append(img_result.summary())
    return "\n".join(lines)


def _list_output_images() -> list[str]:
    """Return paths of all images currently in the output directory."""
    return sorted(
        os.path.join(OUTPUT_DIR, f)
        for f in os.listdir(OUTPUT_DIR)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    )


# ---------------------------------------------------------------------------
# Core processing function
# ---------------------------------------------------------------------------

def process(input_files, progress=gr.Progress(track_tqdm=True)):
    """
    Called by Gradio when the user clicks Run.

    Accepts a list of uploaded file paths from gr.File, copies them into
    INPUT_DIR, runs the pipeline, and returns (log_text, output_gallery).
    """
    if not input_files:
        return "No files uploaded.", []

    # Clear input dir and copy uploaded files in
    for f in os.listdir(INPUT_DIR):
        if f.lower().endswith(SUPPORTED_EXTENSIONS):
            os.remove(os.path.join(INPUT_DIR, f))

    for upload in input_files:
        src = upload if isinstance(upload, str) else upload.name
        dest = os.path.join(INPUT_DIR, os.path.basename(src))
        shutil.copy2(src, dest)

    # Run pipeline
    stats = run(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, models=MODELS)
    log = _format_stats(stats)

    # Collect output images for gallery
    output_images = _list_output_images()

    return log, output_images


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

_EXIFTOOL_NOTE = (
    "⚠️ **exiftool not found** — metadata will not be stripped from output images. "
    "See README for installation instructions."
    if not metadata.is_available()
    else "✅ exiftool found — metadata will be stripped from all output images."
)

with gr.Blocks(title="Refacer", theme=gr.themes.Base()) as demo:
    gr.Markdown(
        """
# Refacer
**Batch face anonymization for photojournalists and activist photographers.**
All processing happens locally — your images never leave this machine.
        """
    )

    gr.Markdown(_EXIFTOOL_NOTE)

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Input images",
                file_count="multiple",
                file_types=list(SUPPORTED_EXTENSIONS),
            )
            run_btn = gr.Button("Run", variant="primary")

        with gr.Column(scale=2):
            log_output = gr.Textbox(
                label="Processing log",
                lines=18,
                interactive=False,
            )

    gallery = gr.Gallery(
        label="Output images",
        columns=4,
        object_fit="contain",
        height="auto",
    )

    run_btn.click(
        fn=process,
        inputs=[file_input],
        outputs=[log_output, gallery],
    )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False,       # never share externally — offline tool
    )