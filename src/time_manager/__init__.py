"""
time-manager: Precision HF Time Transfer Daemon

This package provides a standalone timing service that extracts UTC(NIST)
from WWV/WWVH/CHU time standard broadcasts. It abstracts the ionospheric
channel, providing clean D_clock and Station_ID to any consuming application.

Architecture:
    radiod (RTP) → time-manager → chronyd (SHM) + ZeroMQ/SHM (apps)

The time-manager is INFRASTRUCTURE, not a science application. It provides:
    1. D_clock (system clock offset from UTC)
    2. Station identification on shared frequencies
    3. Propagation mode estimation
    4. Uncertainty quantification

Consumers (like grape-recorder) trust time-manager for timestamps and
station labels, allowing them to focus on their domain-specific tasks.

Version: 1.0.0
Author: Michael James Hauan (AC0G)
"""

__version__ = "1.0.0"
__author__ = "Michael James Hauan (AC0G)"

from .interfaces.timing_result import (
    TimingResult,
    ChannelTimingResult,
    FusionResult,
    DiscriminationInfo,
    ClockStatus,
)

__all__ = [
    "TimingResult",
    "ChannelTimingResult", 
    "FusionResult",
    "DiscriminationInfo",
    "ClockStatus",
    "__version__",
]
