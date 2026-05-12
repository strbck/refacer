"""
refacer.models
~~~~~~~~~~~~~~
Loads and holds references to all ML models used in the pipeline.
Call `load_models(models_dir)` once at startup; pass the returned
`ModelBundle` into the pipeline.  No network I/O happens here —
all weights must already be present on disk.
"""

import logging
import os

logger = logging.getLogger(__name__)

# Model filenames — must match what download_models.py places on disk
INSWAPPER_FILENAME = "inswapper_128.onnx"
GFPGAN_FILENAME = "GFPGANv1.4.pth"


class ModelBundle:
    """Container for all loaded models, passed through the pipeline."""

    def __init__(self, app, swapper, enhancer):
        self.app = app          # insightface FaceAnalysis
        self.swapper = swapper  # inswapper_128 ONNX model
        self.enhancer = enhancer  # GFPGANer


def load_models(models_dir: str) -> ModelBundle:
    """
    Load all required models from *models_dir*.

    Raises
    ------
    FileNotFoundError
        If any required model weight file is missing.
    ImportError
        If a required library is not installed.
    """
    inswapper_path = os.path.join(models_dir, INSWAPPER_FILENAME)
    gfpgan_path = os.path.join(models_dir, GFPGAN_FILENAME)

    # Validate weights exist before importing heavy libraries
    for path, name in [
        (inswapper_path, INSWAPPER_FILENAME),
        (gfpgan_path, GFPGAN_FILENAME),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found: {name}\n"
                f"Expected at: {path}\n"
                "Run scripts/download_models.py to fetch model weights."
            )

    logger.info("Loading FaceAnalysis (buffalo_l)…")
    try:
        from insightface.app import FaceAnalysis
    except ImportError as e:
        raise ImportError("insightface is not installed. Run: pip install insightface") from e

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(1024, 1024))
    logger.info("FaceAnalysis ready.")

    logger.info("Loading inswapper_128…")
    try:
        from insightface.model_zoo import get_model
    except ImportError as e:
        raise ImportError("insightface is not installed. Run: pip install insightface") from e

    swapper = get_model(inswapper_path, providers=["CPUExecutionProvider"])
    logger.info("inswapper_128 ready.")

    logger.info("Loading GFPGANer…")
    try:
        from gfpgan import GFPGANer
    except ImportError as e:
        raise ImportError(
            f"gfpgan failed to import ({e}). Run: pip install gfpgan"
        ) from e

    enhancer = GFPGANer(
        model_path=gfpgan_path,
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )
    logger.info("GFPGANer ready.")

    return ModelBundle(app=app, swapper=swapper, enhancer=enhancer)