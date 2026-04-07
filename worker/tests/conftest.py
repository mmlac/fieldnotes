"""Pytest bootstrap: ensure the tests directory is on sys.path so that
sibling helper modules (e.g. ``_fake_queue``) can be imported by test
files.  This is needed because the tests directory has no
``__init__.py`` and pytest's default import mode does not automatically
add it to ``sys.path``.
"""

from __future__ import annotations

import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
