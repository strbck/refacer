import os
from unittest.mock import MagicMock, patch

import pytest

from refacer.pipeline import (
    FaceResult,
    ImageResult,
    RunStats,
    count_images,
    run,
)
