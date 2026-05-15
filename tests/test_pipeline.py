import os
from unittest.mock import MagicMock, patch

import pytest

from refacer.pipeline import (
    FaceResult,
    ImageResult,
    RunStats,
    # count_images,  # TODO: will be imported in Task 3
    run,
)


class TestRunStatsFromResults:
    def _make_image_result(self, filename, success, faces_detected=0, face_successes=(), enhancement_ok=True, metadata_scrubbed=True):
        face_results = [FaceResult(index=i, success=ok) for i, ok in enumerate(face_successes)]
        return ImageResult(
            filename=filename,
            success=success,
            faces_detected=faces_detected,
            face_results=face_results,
            enhancement_ok=enhancement_ok,
            metadata_scrubbed=metadata_scrubbed,
        )

    def test_empty_results(self):
        stats = RunStats.from_results(total=0, results=[])
        assert stats.total == 0
        assert stats.saved == 0
        assert stats.skipped == 0
        assert stats.failed == 0
        assert stats.total_faces == 0
        assert stats.faces_swapped == 0
        assert stats.faces_failed == 0
        assert stats.image_results == []

    def test_saved_image_counted(self):
        r = self._make_image_result("a.jpg", success=True, faces_detected=2, face_successes=(True, True))
        stats = RunStats.from_results(total=1, results=[r])
        assert stats.saved == 1
        assert stats.skipped == 0
        assert stats.failed == 0
        assert stats.total_faces == 2
        assert stats.faces_swapped == 2
        assert stats.faces_failed == 0

    def test_skipped_image_counted(self):
        r = self._make_image_result("b.jpg", success=True, faces_detected=0)
        stats = RunStats.from_results(total=1, results=[r])
        assert stats.skipped == 1
        assert stats.saved == 0
        assert stats.failed == 0

    def test_failed_image_counted(self):
        r = self._make_image_result("c.jpg", success=False)
        stats = RunStats.from_results(total=1, results=[r])
        assert stats.failed == 1
        assert stats.saved == 0
        assert stats.skipped == 0

    def test_partial_face_swap(self):
        r = self._make_image_result("d.jpg", success=True, faces_detected=3, face_successes=(True, False, True))
        stats = RunStats.from_results(total=1, results=[r])
        assert stats.faces_swapped == 2
        assert stats.faces_failed == 1
        assert stats.saved == 1

    def test_image_results_preserved(self):
        r1 = self._make_image_result("a.jpg", success=True, faces_detected=1, face_successes=(True,))
        r2 = self._make_image_result("b.jpg", success=False)
        stats = RunStats.from_results(total=2, results=[r1, r2])
        assert stats.image_results == [r1, r2]
        assert stats.total == 2

    def test_total_overrides_len_results(self):
        # total is passed explicitly so callers can set it from count_images()
        stats = RunStats.from_results(total=5, results=[])
        assert stats.total == 5
