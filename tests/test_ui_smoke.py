"""Smoke test: the Gradio UI builds without model files or ML libraries."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import gradio as gr
import pytest

_APP_PATH = Path(__file__).parent.parent / "ui" / "app.py"


def _load_ui_app():
    """Execute ui/app.py in a fresh module with load_models stubbed out."""
    sys.modules.pop("ui.app", None)
    spec = importlib.util.spec_from_file_location("ui.app", _APP_PATH)
    module = importlib.util.module_from_spec(spec)
    with patch("refacer.models.load_models", return_value=MagicMock()):
        spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def app():
    return _load_ui_app()


def test_demo_is_gradio_blocks(app):
    assert isinstance(app.demo, gr.Blocks)


def test_demo_has_run_button(app):
    assert isinstance(app.run_btn, gr.Button)


def test_demo_has_file_input(app):
    assert isinstance(app.file_input, gr.File)


def test_demo_has_gallery(app):
    assert isinstance(app.gallery, gr.Gallery)


def test_demo_has_log_output(app):
    assert isinstance(app.log_output, gr.Textbox)
