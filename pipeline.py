"""
refacer.pipeline
~~~~~~~~~~~~~~~~
Resilient batch orchestrator.  Loads images from an input directory,
runs each through the full anonymisation pipeline, and writes results
to an output directory.

Resilience contract
-------------------
- If a single face swap fails, the partially-processed image is saved
  (other faces already swapped) and the failure is logged.
- If face detection fails for a whole image, the original is copied to
  output unchanged and the failure is logged.
- If GFPGAN enhancement fails, the un-enhanced (but swapped) image is
  saved and the failure is logged.
- If metadata scrub or verification fails (and exiftool is present),
  the output is DISCARDED so an unscrubbed image is never published.
- Processing continues to the next image in all cases.

Usage
-----
    from refacer.models import load_models
    from refacer import pipeline

    models = load_models("/path/to/models")
    stats = pipeline.run(
        input_dir="/path/to/input",
        output_dir="/path/to/output",
        models=models,
    )
    print(stats)
"""

import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import List

import cv2

from refacer import metadata
from refacer.models import ModelBundle
from refacer.swap import swap_face

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tiff", ".webp")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class FaceResult:
    """Outcome for a single face within an image."""
    index: int
    success: bool
    error: str = ""


@dataclass
class ImageResult:
    """Outcome for a single image."""
    filename: str
    success: bool                        # True if image was saved (even partially)
    faces_detected: int = 0
    face_results: List[FaceResult] = field(default_factory=list)
    enhancement_ok: bool = True
    metadata_scrubbed: bool = False
    error: str = ""                      # set only on whole-image failure

    @property
    def faces_swapped(self) -> int:
        return sum(1 for f in self.face_results if f.success)

    @property
    def faces_failed(self) -> int:
        return sum(1 for f in self.face_results if not f.success)

    def summary(self) -> str:
        if not self.success:
            return f"  FAILED  {self.filename}: {self.error}"
        parts = [f"  OK      {self.filename}"]
        parts.append(
            f"  faces: {self.faces_swapped}/{self.faces_detected} swapped"
        )
        if self.faces_failed:
            parts.append(f"  ({self.faces_failed} face(s) failed — partial save)")
        if not self.enhancement_ok:
            parts.append("  enhancement failed — saved without GFPGAN")
        if metadata.is_available() and not self.metadata_scrubbed:
            parts.append("  WARNING: metadata scrub failed")
        elif not metadata.is_available():
            parts.append("  metadata scrub skipped (exiftool not installed)")
        return "\n".join(parts)


@dataclass
class RunStats:
    """Aggregate statistics for a full pipeline run."""
    total: int = 0
    saved: int = 0
    skipped: int = 0          # no faces detected — original copied
    failed: int = 0           # could not be saved at all
    total_faces: int = 0
    faces_swapped: int = 0
    faces_failed: int = 0
    image_results: List[ImageResult] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "",
            "── Refacer run complete ──────────────────────────────────────",
            f"  Images processed : {self.total}",
            f"  Saved            : {self.saved}",
            f"  No faces (copied): {self.skipped}",
            f"  Failed entirely  : {self.failed}",
            f"  Faces detected   : {self.total_faces}",
            f"  Faces swapped    : {self.faces_swapped}",
            f"  Faces failed     : {self.faces_failed}",
            "──────────────────────────────────────────────────────────────",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _collect_images(input_dir: str) -> List[str]:
    return sorted(
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    )


def _scrub_verify_finalize(
    temp_path: str,
    final_path: str,
    filename: str,
) -> tuple[bool, bool]:
    """
    Scrub metadata from *temp_path*, verify the result is clean, and
    atomically promote it to *final_path*.

    If exiftool is not available the file is moved into place without
    scrubbing or verification (consistent with the resilience contract:
    the pipeline still produces output, just without the metadata
    guarantee — a prominent warning is logged at startup).

    If exiftool is available and either scrubbing or verification fails,
    *temp_path* is deleted and ``(False, False)`` is returned. We refuse
    to publish an output image whose metadata could not be guaranteed.

    Returns
    -------
    (saved, scrubbed) : tuple[bool, bool]
        saved    — True if the file ended up at *final_path*.
        scrubbed — True if metadata was both scrubbed and verified clean.
                   Always False when exiftool is unavailable.
    """
    if metadata.is_available():
        scrubbed = metadata.scrub(temp_path)
        verified = scrubbed and metadata.verify_scrubbed(temp_path)
        if not verified:
            logger.error(
                "%s — metadata scrub/verify FAILED. "
                "Output discarded to prevent leaking identifying metadata.",
                filename,
            )
            try:
                os.remove(temp_path)
            except OSError as exc:
                logger.warning(
                    "%s — could not clean up temp file %s: %s",
                    filename, temp_path, exc,
                )
            return False, False
        scrubbed_ok = True
    else:
        scrubbed_ok = False

    try:
        os.replace(temp_path, final_path)
    except OSError as exc:
        logger.error(
            "%s — could not move output into place: %s", filename, exc
        )
        try:
            os.remove(temp_path)
        except OSError:
            pass
        return False, scrubbed_ok

    return True, scrubbed_ok


def _process_image(
    filename: str,
    input_dir: str,
    output_dir: str,
    models: ModelBundle,
) -> ImageResult:
    """Run the full pipeline for a single image file."""
    result = ImageResult(filename=filename, success=False)
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    # --- Load ---
    img = cv2.imread(input_path)
    if img is None:
        result.error = "cv2.imread returned None — unreadable or corrupt file"
        logger.error("Could not read %s, skipping.", filename)
        return result

    # --- Detect faces ---
    try:
        faces = models.app.get(img)
    except Exception as exc:  # noqa: BLE001
        result.error = f"Face detection failed: {exc}"
        logger.error("Face detection failed on %s: %s", filename, exc)
        return result

    result.faces_detected = len(faces)
    logger.info("%s — %d face(s) detected", filename, len(faces))

    # No faces: copy original to output unchanged (still scrub + verify)
    if not faces:
        logger.info("%s — no faces detected, copying original to output.", filename)
        temp_path = "{}.tmp{}".format(*os.path.splitext(output_path))
        try:
            shutil.copy2(input_path, temp_path)
        except OSError as exc:
            result.error = f"Could not copy original to temp: {exc}"
            logger.error("Failed to copy %s to temp: %s", filename, exc)
            return result

        saved, scrubbed = _scrub_verify_finalize(temp_path, output_path, filename)
        if not saved:
            result.error = (
                "Metadata scrub/verify failed — output discarded to "
                "prevent leaking identifying metadata."
            )
            return result
        result.success = True
        result.metadata_scrubbed = scrubbed
        return result

    # --- Swap faces (resilient per-face) ---
    current = img.copy()
    for i, face in enumerate(faces):
        try:
            current = swap_face(models.swapper, current, face)
            result.face_results.append(FaceResult(index=i, success=True))
            logger.debug("%s — face %d swapped OK", filename, i)
        except Exception as exc:  # noqa: BLE001
            result.face_results.append(FaceResult(index=i, success=False, error=str(exc)))
            logger.warning(
                "%s — face %d swap FAILED (partial save): %s", filename, i, exc
            )

    # --- GFPGAN enhancement ---
    try:
        _, _, current = models.enhancer.enhance(
            current,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )
        logger.debug("%s — enhancement OK", filename)
    except Exception as exc:  # noqa: BLE001
        result.enhancement_ok = False
        logger.warning("%s — GFPGAN enhancement failed, saving without: %s", filename, exc)

    # --- Write to temp, scrub + verify, then atomically promote ---
    temp_path = "{}.tmp{}".format(*os.path.splitext(output_path))
    if not cv2.imwrite(temp_path, current):
        result.error = f"cv2.imwrite failed — could not write to {temp_path}"
        logger.error("Failed to write output for %s", filename)
        return result

    saved, scrubbed = _scrub_verify_finalize(temp_path, output_path, filename)
    if not saved:
        result.error = (
            "Metadata scrub/verify failed — output discarded to "
            "prevent leaking identifying metadata."
        )
        return result

    result.success = True
    result.metadata_scrubbed = scrubbed
    logger.info("%s — saved to output/", filename)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    input_dir: str,
    output_dir: str,
    models: ModelBundle,
) -> RunStats:
    """
    Process all supported images in *input_dir*, writing results to *output_dir*.

    Parameters
    ----------
    input_dir  : str  — path to folder containing source images
    output_dir : str  — path to folder for anonymised output images
    models     : ModelBundle — pre-loaded models from refacer.models.load_models()

    Returns
    -------
    RunStats with per-image results and aggregate counts.
    """
    os.makedirs(output_dir, exist_ok=True)

    filenames = _collect_images(input_dir)
    if not filenames:
        logger.warning("No supported images found in %s", input_dir)
        return RunStats()

    logger.info("Found %d image(s) to process in %s", len(filenames), input_dir)

    stats = RunStats(total=len(filenames))

    for filename in filenames:
        logger.info("── Processing: %s", filename)
        image_result = _process_image(filename, input_dir, output_dir, models)
        stats.image_results.append(image_result)

        stats.total_faces += image_result.faces_detected
        stats.faces_swapped += image_result.faces_swapped
        stats.faces_failed += image_result.faces_failed

        if not image_result.success:
            stats.failed += 1
        elif image_result.faces_detected == 0:
            stats.skipped += 1
        else:
            stats.saved += 1

        logger.info(image_result.summary())

    return stats
