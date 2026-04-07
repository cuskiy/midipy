"""Render pipeline: per-track synthesis → mix → master → FLAC.

Public API re-exported from sub-modules:
  render        – top-level MIDI → FLAC
  clear_caches  – reset all pipeline caches
  BLOCK_SIZE    – block size constant (used by tests)
  _render_track – (testing only) single-track render
"""
from __future__ import annotations

from config import BLOCK_SIZE
from .pipeline import render, clear_caches
from .track_render import render_track as _render_track

__all__ = ["render", "clear_caches", "BLOCK_SIZE", "_render_track"]
