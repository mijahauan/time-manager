"""
Timing Result Data Models

These dataclasses define the contract between time-manager and its consumers.
The TimingResult is serialized to JSON and written to shared memory for
consumption by grape-recorder and other applications.

Contract Version: 1.0.0
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, List
import json
import time


class ClockStatus(str, Enum):
    """Clock synchronization status."""
    ACQUIRING = "ACQUIRING"   # Initial startup, collecting data
    LOCKED = "LOCKED"         # Stable, high-confidence timing
    HOLDOVER = "HOLDOVER"     # Lost lock, using last known state
    UNLOCKED = "UNLOCKED"     # No timing available


@dataclass
class DiscriminationInfo:
    """
    Station discrimination result for a shared frequency.
    
    On shared frequencies (2.5, 5, 10, 15 MHz), both WWV and WWVH transmit.
    This structure captures how the station was identified.
    """
    method: str                          # 'ground_truth_500hz', 'bcd_correlation', 'power_ratio', etc.
    power_ratio_db: Optional[float] = None         # WWV - WWVH power (positive = WWV stronger)
    differential_delay_ms: Optional[float] = None  # WWV - WWVH arrival time
    bcd_correlation_quality: Optional[float] = None
    ground_truth_minute: Optional[int] = None      # If detected during exclusive minute
    confidence_score: float = 0.0                  # 0-1 overall confidence
    
    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ChannelTimingResult:
    """
    Timing result for a single channel (frequency).
    
    Each channel produces independent timing measurements that are later
    fused across all channels for improved accuracy.
    """
    channel_name: str                    # "WWV 10 MHz", "CHU 3.33 MHz"
    station: str                         # "WWV", "WWVH", "CHU", "UNKNOWN"
    confidence: str                      # "high", "medium", "low"
    
    # Timing measurements
    tone_detected: bool = False
    timing_error_ms: Optional[float] = None      # Offset from expected second boundary
    propagation_delay_ms: Optional[float] = None # Calculated propagation delay
    d_clock_raw_ms: Optional[float] = None       # Raw D_clock before calibration
    
    # Propagation mode
    propagation_mode: str = "UNKNOWN"            # "1F2", "2F2", "GW", etc.
    n_hops: int = 0
    layer_height_km: float = 0.0
    
    # Signal quality
    snr_db: Optional[float] = None
    carrier_power_db: Optional[float] = None
    
    # Station discrimination (only for shared frequencies)
    discrimination: Optional[DiscriminationInfo] = None
    is_shared_frequency: bool = False
    
    # Uncertainty
    uncertainty_ms: float = 10.0
    
    def to_dict(self) -> dict:
        result = asdict(self)
        if self.discrimination:
            result['discrimination'] = self.discrimination.to_dict()
        return {k: v for k, v in result.items() if v is not None}


@dataclass
class FusionResult:
    """
    Result of multi-broadcast fusion across all channels.
    
    Combines measurements from up to 13 broadcasts (9 frequencies, with
    WWV+WWVH on 4 shared frequencies) to achieve sub-millisecond accuracy.
    """
    contributing_broadcasts: int         # Number of broadcasts used in fusion
    total_broadcasts: int                # Total possible (typically 13)
    
    # Fused D_clock
    fused_d_clock_ms: float
    fusion_uncertainty_ms: float
    
    # Per-station calibration offsets
    wwv_calibration_offset_ms: float = 0.0
    wwvh_calibration_offset_ms: float = 0.0
    chu_calibration_offset_ms: float = 0.0
    
    # Quality metrics
    chi_squared: Optional[float] = None  # Goodness of fit
    outlier_count: int = 0               # Broadcasts rejected as outliers
    
    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class TimingResult:
    """
    Complete timing result published by time-manager.
    
    This is the top-level structure written to shared memory and consumed
    by grape-recorder and other applications. It represents the timing
    state at a specific instant.
    """
    # Version for contract compatibility
    version: str = "1.0.0"
    
    # Timestamps
    timestamp: float = 0.0               # UTC timestamp this result represents
    system_time: float = 0.0             # System clock at measurement time
    generated_at: float = field(default_factory=time.time)
    
    # Primary output: D_clock
    d_clock_ms: float = 0.0              # System clock - UTC(NIST)
    d_clock_uncertainty_ms: float = 10.0
    
    # Clock status
    clock_status: ClockStatus = ClockStatus.ACQUIRING
    
    # Fusion result (combines all channels)
    fusion: Optional[FusionResult] = None
    
    # Per-channel results
    channels: Dict[str, ChannelTimingResult] = field(default_factory=dict)
    
    # System health
    channels_active: int = 0
    channels_locked: int = 0
    uptime_seconds: float = 0.0
    
    def to_json(self) -> str:
        """Serialize to JSON for shared memory or ZeroMQ."""
        data = {
            "version": self.version,
            "timestamp": self.timestamp,
            "system_time": self.system_time,
            "generated_at": self.generated_at,
            "d_clock_ms": self.d_clock_ms,
            "d_clock_uncertainty_ms": self.d_clock_uncertainty_ms,
            "clock_status": self.clock_status.value,
            "channels_active": self.channels_active,
            "channels_locked": self.channels_locked,
            "uptime_seconds": self.uptime_seconds,
        }
        
        if self.fusion:
            data["fusion"] = self.fusion.to_dict()
        
        data["channels"] = {
            name: ch.to_dict() for name, ch in self.channels.items()
        }
        
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "TimingResult":
        """Deserialize from JSON."""
        data = json.loads(json_str)
        
        result = cls(
            version=data.get("version", "1.0.0"),
            timestamp=data.get("timestamp", 0.0),
            system_time=data.get("system_time", 0.0),
            generated_at=data.get("generated_at", time.time()),
            d_clock_ms=data.get("d_clock_ms", 0.0),
            d_clock_uncertainty_ms=data.get("d_clock_uncertainty_ms", 10.0),
            clock_status=ClockStatus(data.get("clock_status", "ACQUIRING")),
            channels_active=data.get("channels_active", 0),
            channels_locked=data.get("channels_locked", 0),
            uptime_seconds=data.get("uptime_seconds", 0.0),
        )
        
        # Reconstruct fusion
        if "fusion" in data:
            f = data["fusion"]
            result.fusion = FusionResult(
                contributing_broadcasts=f.get("contributing_broadcasts", 0),
                total_broadcasts=f.get("total_broadcasts", 13),
                fused_d_clock_ms=f.get("fused_d_clock_ms", 0.0),
                fusion_uncertainty_ms=f.get("fusion_uncertainty_ms", 10.0),
            )
        
        # Reconstruct channels (simplified - full reconstruction would rebuild all nested objects)
        for name, ch_data in data.get("channels", {}).items():
            disc_data = ch_data.pop("discrimination", None)
            disc = None
            if disc_data:
                disc = DiscriminationInfo(**disc_data)
            
            result.channels[name] = ChannelTimingResult(
                channel_name=ch_data.get("channel_name", name),
                station=ch_data.get("station", "UNKNOWN"),
                confidence=ch_data.get("confidence", "low"),
                discrimination=disc,
                **{k: v for k, v in ch_data.items() 
                   if k not in ("channel_name", "station", "confidence")}
            )
        
        return result
