"""Sphinx configuration for the PCP EEG Pipeline documentation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "PCP EEG Pipeline"
author = "PCP EEG Project 2026"
copyright = "2026, PCP EEG Project 2026"

from pcp_project import __version__  # noqa: E402

version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autodoc_default_options = {
    "members": True,
    "exclude-members": (
        "set_decision_function_request,set_fit_request,set_predict_request,"
        "set_predict_log_proba_request,set_predict_proba_request,"
        "set_score_request,set_transform_request"
    ),
}
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path: list[str] = []

html_theme = "furo"
html_title = f"{project} {release}"
html_static_path: list[str] = []
