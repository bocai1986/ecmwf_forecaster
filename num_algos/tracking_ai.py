"""Compatibility shim for users expecting ``num_algos.tracking_ai`` at repo root.

The implementation lives in :mod:`offline.num_algos.tracking_ai`; this module simply
re-exports all public symbols so existing import paths keep working and the file can
also be downloaded directly via ``num_algos/tracking_ai.py``.
"""

from offline.num_algos.tracking_ai import *  # noqa: F401,F403
