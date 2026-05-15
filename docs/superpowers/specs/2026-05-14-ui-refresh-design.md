# UI Refresh & Live Streaming Pipeline — Design Spec

**Date:** 2026-05-14
**Status:** Approved

---

## Overview

Refactor the Gradio UI into a clean three-panel layout and refactor `pipeline.run()` into a generator so the UI can stream live log updates and gallery additions as each image finishes processing. No network calls are introduced; all processing remains fully offline.

---

## Goals

- Replace the current side-by-side layout with a vertical three-panel flow: Upload → Progress → Output.
- Stream per-image results to the browser in real time (log lines, stat cards, gallery thumbnails) as the pipeline processes each file.
- Keep the CLI working with identical user-facing output and live terminal logs.

---

## Architecture

### 1. Pipeline generator API (`pipeline.py`)

`run()` changes from returning a `RunStats` to being a generator that `yield`s one `ImageResult` per image as it finishes.

```python
def run(
    input_dir: str,
    output_dir: str,
    models: ModelBundle,
) -> Generator[ImageResult, None, None]:
    filenames = _collect_images(input_dir)
    for filename in filenames:
        yield _process_image(filename, input_dir, output_dir, models)
        gc.collect()
```

**New public helpers added to `pipeline.py`:**

- `count_images(input_dir: str) -> int` — returns the number of supported images in a directory without consuming the generator. Used by callers to know `total` upfront for progress display.
- `RunStats.from_results(total: int, results: list[ImageResult]) -> RunStats` — classmethod that builds the aggregate from an already-accumulated list of results. Replaces the scattered `stats.saved += 1` / `stats.total_faces += ...` logic that was previously spread through `run()`.

**Unchanged:** `_process_image`, `_scrub_verify_finalize`, `_collect_images`, all resilience behaviour, and all data classes (`FaceResult`, `ImageResult`, `RunStats`).

### 2. CLI adapter (`__main__.py`)

The CLI iterates the generator, accumulates results, then builds and prints `RunStats` at the end. Since `_process_image` already calls `logger.info` / `logger.warning` throughout, the terminal sees live log output as each image finishes — same behaviour as today.

```python
results = []
for img_result in pipeline.run(input_dir, output_dir, models):
    results.append(img_result)

stats = RunStats.from_results(len(results), results)
print(stats)
for r in stats.image_results:
    print(r.summary())
```

No changes to CLI flags, argument parsing, or terminal output format.

### 3. UI layout (`ui/app.py`)

Three stacked panels inside `gr.Blocks(title="Refacer", theme=gr.themes.Base())`.

#### Panel 1 — Upload

| Component | Config |
|---|---|
| `gr.File` | `file_count="multiple"`, `file_types=list(SUPPORTED_EXTENSIONS)` |
| `gr.Button("Run")` | `variant="primary"` |
| `gr.Button("Clear")` | resets file input and clears output dir |
| `gr.Textbox` (file count label) | `interactive=False`, single line, shows e.g. `"7 images selected"` |

#### Panel 2 — Progress

| Component | Config |
|---|---|
| `gr.Textbox` (log stream) | `lines=10`, `interactive=False`, `autoscroll=True` |
| `gr.Progress` | built-in Gradio loading indicator on the Run button |
| `gr.Number` × 3 | `interactive=False`; labelled **Done**, **Faces Swapped**, **Warnings** |

#### Panel 3 — Output

| Component | Config |
|---|---|
| `gr.Gallery` | `columns=4`, `object_fit="contain"`, `height="auto"` |
| `gr.Button("Open output folder")` | calls `os.startfile` / `open` on the output directory |

### 4. Streaming in `process()`

`process()` is a Gradio generator. Outputs tuple: `(log_text, gallery_images, done_count, faces_swapped_count, warnings_count)`.

```python
def process(input_files, progress=gr.Progress()):
    if not input_files:
        yield "No files uploaded.", [], 0, 0, 0
        return

    # Clear input dir, copy uploads in
    _clear_and_copy(input_files)

    total = pipeline.count_images(INPUT_DIR)
    results = []

    for img_result in pipeline.run(INPUT_DIR, OUTPUT_DIR, MODELS):
        results.append(img_result)
        progress(len(results) / total)
        yield (
            _build_log(results),
            _list_output_images(),
            len(results),
            sum(r.faces_swapped for r in results),
            _warning_count(results),
        )

    # Final summary appended to log
    stats = RunStats.from_results(total, results)
    yield (
        _build_log(results) + "\n" + str(stats),
        _list_output_images(),
        total,
        stats.faces_swapped,
        _warning_count(results),
    )
```

`_build_log(results)` concatenates `r.summary()` for each result with newlines. `_list_output_images()` rescans the output directory after each image is promoted (same helper as today). `_warning_count(results)` counts the number of images that had any warning condition (partial face swap, enhancement failure, or metadata scrub failure) — consistent across streaming yields and the final yield. `_clear_and_copy(input_files)` is the existing input-dir clearing + file copying logic extracted from the current `process()` into a named helper.

**Open output folder button:** Uses `subprocess.run(["open", OUTPUT_DIR])` on macOS, `subprocess.run(["xdg-open", OUTPUT_DIR])` on Linux, and `os.startfile(OUTPUT_DIR)` on Windows, selected via `sys.platform`.

### 5. Error handling

No new error categories are introduced. The existing resilience contract is preserved end-to-end:

- Per-face swap failure → partial save, logged, appears in Warnings count.
- Whole-image detection failure → original copied to output, appears in Done count (not Warnings).
- GFPGAN failure → saved without enhancement, appears in Warnings count.
- Metadata scrub/verify failure → output discarded, appears in Warnings count; log line makes this explicit.

If `process()` raises an unexpected exception (e.g. `ModelBundle` not loaded), Gradio surfaces it as an error banner. This is unchanged from today.

---

## File Change Summary

| File | Change |
|---|---|
| `pipeline.py` | `run()` → generator; add `count_images()`; add `RunStats.from_results()` |
| `__main__.py` | adapt loop to iterate generator; build `RunStats` at end |
| `ui/app.py` | full layout rewrite to 3-panel design; `process()` → generator |

No new dependencies. No changes to `swap.py`, `models.py`, `metadata.py`, or Docker/CI config.

---

## Out of Scope

- Real-time per-face progress within a single image (would require refactoring `_process_image`).
- "Download all" as a zip (no file archiving logic added).
- Dark/light theme toggle.
- Any changes to model loading, face detection, or swap logic.
