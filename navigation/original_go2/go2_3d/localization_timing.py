"""Shared receipt deadline for the mapper's 10 Hz localization heartbeat.

A short delivery/scheduling gap may retain the previous valid state. Explicit
invalid reports are never debounced, and this is independent of scan/odometry
freshness, bounded dead reckoning, TF freshness and the command watchdog.
"""
import math

LOCALIZATION_HEARTBEAT_TIMEOUT = 1.0


def localization_is_fresh(valid, received, now):
    """Require affirmative validity and a finite, recent monotonic receipt."""
    return (bool(valid) and received is not None and
            math.isfinite(received) and math.isfinite(now) and
            0. <= now - received < LOCALIZATION_HEARTBEAT_TIMEOUT)
