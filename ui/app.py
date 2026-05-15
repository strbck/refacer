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
import subprocess
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
from refacer.pipeline import SUPPORTED_EXTENSIONS, RunStats, count_images, run

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

def _build_log(results: list) -> str:
    return "\n".join(r.summary() for r in results)


def _warning_count(results: list) -> int:
    return sum(
        1 for r in results
        if r.faces_failed > 0
        or not r.enhancement_ok
        or (metadata.is_available() and r.success and not r.metadata_scrubbed)
    )


def _list_output_images() -> list[str]:
    return sorted(
        os.path.join(OUTPUT_DIR, f)
        for f in os.listdir(OUTPUT_DIR)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    )


def _clear_and_copy(input_files) -> None:
    for f in os.listdir(INPUT_DIR):
        if f.lower().endswith(SUPPORTED_EXTENSIONS):
            os.remove(os.path.join(INPUT_DIR, f))
    for upload in input_files:
        src = upload if isinstance(upload, str) else upload.name
        dest = os.path.join(INPUT_DIR, os.path.basename(src))
        if os.path.exists(dest):
            logger.warning("Duplicate basename %s — earlier file overwritten", os.path.basename(src))
        shutil.copy2(src, dest)


# ---------------------------------------------------------------------------
# Core processing function
# ---------------------------------------------------------------------------

def process(input_files, progress=gr.Progress()):
    if not input_files:
        yield "No files uploaded.", [], 0, 0, 0
        return

    _clear_and_copy(input_files)

    total = count_images(INPUT_DIR)
    if total == 0:
        yield "No supported images found in upload.", [], 0, 0, 0
        return

    results = []
    for img_result in run(INPUT_DIR, OUTPUT_DIR, MODELS):
        results.append(img_result)
        progress(len(results) / total)
        yield (
            _build_log(results),
            _list_output_images(),
            len(results),
            sum(r.faces_swapped for r in results),
            _warning_count(results),
        )

    stats = RunStats.from_results(total, results)
    yield (
        _build_log(results) + "\n" + str(stats),
        _list_output_images(),
        total,
        stats.faces_swapped,
        _warning_count(results),
    )


def clear_inputs():
    return None, "", [], 0, 0, 0


def open_output_folder():
    if sys.platform == "darwin":
        subprocess.run(["open", OUTPUT_DIR], check=False)
    elif sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", OUTPUT_DIR], check=False)
    elif sys.platform == "win32":
        os.startfile(OUTPUT_DIR)
    else:
        logger.warning("open_output_folder: unsupported platform %s", sys.platform)


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

    # ── Panel 1: Upload ────────────────────────────────────────────────────
    with gr.Group():
        gr.Markdown("### 1 · Upload")
        file_input = gr.File(
            label="Input images",
            file_count="multiple",
            file_types=list(SUPPORTED_EXTENSIONS),
        )
        with gr.Row():
            run_btn = gr.Button("Run", variant="primary")
            clear_btn = gr.Button("Clear")

    # ── Panel 2: Progress ──────────────────────────────────────────────────
    with gr.Group():
        gr.Markdown("### 2 · Progress")
        log_output = gr.Textbox(
            label="Processing log",
            lines=10,
            interactive=False,
            autoscroll=True,
        )
        with gr.Row():
            done_num = gr.Number(label="Done", value=0, interactive=False)
            faces_num = gr.Number(label="Faces Swapped", value=0, interactive=False)
            warnings_num = gr.Number(label="Warnings", value=0, interactive=False)

    # ── Panel 3: Output ────────────────────────────────────────────────────
    with gr.Group():
        gr.Markdown("### 3 · Output")
        gallery = gr.Gallery(
            label="Output images",
            columns=4,
            object_fit="contain",
            height="auto",
        )
        open_btn = gr.Button("Open output folder")

    run_btn.click(
        fn=process,
        inputs=[file_input],
        outputs=[log_output, gallery, done_num, faces_num, warnings_num],
    )
    clear_btn.click(
        fn=clear_inputs,
        inputs=[],
        outputs=[file_input, log_output, gallery, done_num, faces_num, warnings_num],
    )
    open_btn.click(fn=open_output_folder, inputs=[], outputs=[])

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    # Bind to all interfaces when running inside Docker; localhost otherwise.
    host = "0.0.0.0" if os.environ.get("REFACER_DOCKER") else "127.0.0.1"
    demo.launch(
        server_name=host,
        server_port=7860,
        inbrowser=not os.environ.get("REFACER_DOCKER"),
        share=False,       # never share externally — offline tool
    )