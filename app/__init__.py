"""
babel.app
=========
Gradio UI sub-package for the Babel dubbing pipeline.

Phase 0: stub app with @spaces.GPU pattern wired but pipeline not yet connected.
Phase 4: real pipeline wired into the Gradio UI.

Public exports:
  create_app()  — build and return the Gradio Blocks interface
  launch_app()  — launch the app on localhost (used by Docker CMD)
"""

from babel.app.app import create_app
from babel.app.launch import main as launch_app

__all__ = ["create_app", "launch_app"]
