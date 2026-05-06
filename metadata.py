"""
refacer.metadata
~~~~~~~~~~~~~~~~
Strips all EXIF, XMP, and IPTC metadata from image files using exiftool.

exiftool is a system dependency (not a pip package).  If it is not found
on PATH this module logs a prominent warning and skips scrubbing rather
than crashing — consistent with the pipeline's resilient design.

Installation:
  macOS:   brew install exiftool
  Linux:   sudo apt install libimage-exiftool-perl
  Windows: https://exiftool.org  (add to system PATH)
"""

import json
import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)

# Tag groups that must be empty after a successful scrub. Anything reported
# here is considered identifying metadata for the purposes of verification.
_IDENTIFYING_GROUPS = (
    "EXIF",
    "IPTC",
    "XMP",
    "GPS",
    "MakerNotes",
)

# Checked once at import time so the warning surfaces early.
_EXIFTOOL_AVAILABLE: bool = shutil.which("exiftool") is not None

if not _EXIFTOOL_AVAILABLE:
    logger.warning(
        "┌─────────────────────────────────────────────────────────────────┐\n"
        "│  WARNING: exiftool not found on PATH — metadata will NOT be     │\n"
        "│  stripped from output images.  GPS coordinates, timestamps,     │\n"
        "│  and device identifiers may remain in output files.             │\n"
        "│                                                                 │\n"
        "│  Install exiftool and re-run to enable metadata scrubbing:     │\n"
        "│    macOS:   brew install exiftool                               │\n"
        "│    Linux:   sudo apt install libimage-exiftool-perl             │\n"
        "│    Windows: https://exiftool.org                                │\n"
        "└─────────────────────────────────────────────────────────────────┘"
    )


def is_available() -> bool:
    """Return True if exiftool is installed and on PATH."""
    return _EXIFTOOL_AVAILABLE


def scrub(image_path: str) -> bool:
    """
    Strip all metadata from *image_path* in-place using exiftool.

    Parameters
    ----------
    image_path : str
        Path to the image file to scrub.

    Returns
    -------
    bool
        True if scrubbing succeeded (or was skipped due to missing exiftool).
        False if exiftool was found but returned a non-zero exit code.

    Notes
    -----
    - Operates in-place; exiftool writes to a temp file and renames.
    - The original file with ``_original`` suffix left by exiftool is
      deleted automatically via the ``-overwrite_original`` flag.
    - If exiftool is not available this function logs a warning and
      returns True (skip, not failure) so the pipeline continues.
    """
    if not _EXIFTOOL_AVAILABLE:
        logger.debug("Skipping metadata scrub (exiftool not available): %s", image_path)
        return True

    try:
        result = subprocess.run(
            [
                "exiftool",
                "-all=",                # remove all metadata
                "-overwrite_original",  # no _original backup file
                image_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            logger.error(
                "exiftool failed on %s (exit %d): %s",
                image_path,
                result.returncode,
                result.stderr.strip(),
            )
            return False

        logger.debug("Metadata scrubbed: %s", image_path)
        return True

    except subprocess.TimeoutExpired:
        logger.error("exiftool timed out on %s", image_path)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error running exiftool on %s: %s", image_path, exc)
        return False


def verify_scrubbed(image_path: str) -> bool:
    """
    Verify that *image_path* contains no identifying metadata.

    Runs exiftool in read-only mode and inspects the resulting tag groups.
    Any tag from EXIF, IPTC, XMP, GPS, or MakerNotes is treated as
    identifying metadata and causes verification to fail.

    Parameters
    ----------
    image_path : str
        Path to the image file to inspect.

    Returns
    -------
    bool
        True if no identifying metadata is present, OR if exiftool is not
        available on PATH (verification is skipped, not failed, so the
        pipeline can still operate without exiftool installed).
        False if exiftool found identifying tags, errored out, or its
        output could not be parsed.

    Notes
    -----
    The ``File`` and ``Composite`` groups are deliberately ignored — File
    is metadata about the file itself (size, MIME type, image dimensions)
    and Composite tags are derived from other tags, so they will be empty
    once the underlying identifying tags are gone.
    """
    if not _EXIFTOOL_AVAILABLE:
        logger.debug("Skipping metadata verify (exiftool not available): %s", image_path)
        return True

    try:
        result = subprocess.run(
            [
                "exiftool",
                "-G",          # prefix tag names with group, e.g. "EXIF:Make"
                "-j",          # JSON output
                "-a",          # allow duplicate tags
                "-u",          # include unknown tags
                image_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            logger.error(
                "exiftool verify failed on %s (exit %d): %s",
                image_path,
                result.returncode,
                result.stderr.strip(),
            )
            return False

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            logger.error(
                "Could not parse exiftool output for %s: %s", image_path, exc
            )
            return False

        if not data:
            return True

        tags = data[0]
        leaked = [
            key for key in tags
            if ":" in key and key.split(":", 1)[0] in _IDENTIFYING_GROUPS
        ]

        if leaked:
            logger.warning(
                "%s — verification found %d identifying tag(s) still present: %s",
                image_path,
                len(leaked),
                ", ".join(leaked[:5]) + ("…" if len(leaked) > 5 else ""),
            )
            return False

        logger.debug("Metadata verified clean: %s", image_path)
        return True

    except subprocess.TimeoutExpired:
        logger.error("exiftool verify timed out on %s", image_path)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error verifying %s: %s", image_path, exc)
        return False
