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


class TestCountImages:
    def test_counts_supported_extensions(self, tmp_path):
        (tmp_path / "a.jpg").touch()
        (tmp_path / "b.png").touch()
        (tmp_path / "c.webp").touch()
        (tmp_path / "d.txt").touch()   # excluded
        assert count_images(str(tmp_path)) == 3

    def test_case_insensitive(self, tmp_path):
        (tmp_path / "a.JPG").touch()
        (tmp_path / "b.JPEG").touch()
        (tmp_path / "c.PNG").touch()
        assert count_images(str(tmp_path)) == 3

    def test_empty_directory(self, tmp_path):
        assert count_images(str(tmp_path)) == 0

    def test_no_supported_files(self, tmp_path):
        (tmp_path / "readme.txt").touch()
        (tmp_path / "data.csv").touch()
        assert count_images(str(tmp_path)) == 0


class TestRunGenerator:
    def _fake_result(self, filename, success=True, faces_detected=1):
        return ImageResult(
            filename=filename,
            success=success,
            faces_detected=faces_detected,
            face_results=[FaceResult(0, True)] if faces_detected else [],
        )

    def test_run_yields_image_result_per_image(self, tmp_path):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        (input_dir / "a.jpg").touch()
        (input_dir / "b.jpg").touch()

        r1 = self._fake_result("a.jpg")
        r2 = self._fake_result("b.jpg")

        with patch("refacer.pipeline._process_image", side_effect=[r1, r2]):
            results = list(run(str(input_dir), str(tmp_path / "out"), MagicMock()))

        assert results == [r1, r2]

    def test_run_yields_in_sorted_order(self, tmp_path):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        for name in ["c.jpg", "a.jpg", "b.jpg"]:
            (input_dir / name).touch()

        fake = [self._fake_result(n) for n in ["a.jpg", "b.jpg", "c.jpg"]]

        with patch("refacer.pipeline._process_image", side_effect=fake):
            results = list(run(str(input_dir), str(tmp_path / "out"), MagicMock()))

        assert [r.filename for r in results] == ["a.jpg", "b.jpg", "c.jpg"]

    def test_run_empty_directory_yields_nothing(self, tmp_path):
        input_dir = tmp_path / "in"
        input_dir.mkdir()

        results = list(run(str(input_dir), str(tmp_path / "out"), MagicMock()))

        assert results == []

    def test_run_creates_output_dir(self, tmp_path):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        output_dir = tmp_path / "out"  # does not exist yet

        list(run(str(input_dir), str(output_dir), MagicMock()))

        assert output_dir.exists()

    def test_run_is_a_generator(self, tmp_path):
        import types
        input_dir = tmp_path / "in"
        input_dir.mkdir()

        gen = run(str(input_dir), str(tmp_path / "out"), MagicMock())

        assert isinstance(gen, types.GeneratorType)
