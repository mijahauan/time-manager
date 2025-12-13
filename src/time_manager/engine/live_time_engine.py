#!/usr/bin/env python3
"""
Live Time Engine - Twin-Stream Architecture

This is the core of time-manager's live operation. It subscribes directly
to the RTP multicast stream (parallel to grape-recorder) and performs
real-time timing extraction.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                      RTP Multicast Stream                        │
    │                              │                                   │
    │              ┌───────────────┴───────────────┐                   │
    │              ▼                               ▼                   │
    │    ┌─────────────────┐             ┌─────────────────┐          │
    │    │  time-manager   │             │  grape-recorder │          │
    │    │  (RAM only)     │             │  (disk archive) │          │
    │    └─────────────────┘             └─────────────────┘          │
    └─────────────────────────────────────────────────────────────────┘

Processing Loops:
    FAST LOOP (T=:01): Low-latency D_clock using previous minute's state
        - Ring buffer keeps only ±1.5s around minute boundary
        - Tone detection → D_clock → Chrony SHM
        - Latency: ~1 second after minute

    SLOW LOOP (T=:02): Full discrimination and state update
        - Full minute buffer in RAM
        - BCD correlation, Doppler, FSS → Station/Mode identification
        - Updates state for next Fast Loop
        - Publishes /dev/shm/grape_timing.json
"""

import numpy as np
import logging
import threading
import time
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional, List, Any, Callable
from dataclasses import dataclass, field, asdict
from collections import deque
from enum import Enum

logger = logging.getLogger('time-manager.engine')


class EngineState(Enum):
    """Time engine operational state."""
    STARTING = "STARTING"      # Initializing, no output
    ACQUIRING = "ACQUIRING"    # Building first minute, no Chrony output
    TRACKING = "TRACKING"      # Steady state, outputting to Chrony
    HOLDOVER = "HOLDOVER"      # Lost signal, using predicted state


@dataclass
class ChannelState:
    """Per-channel state used by Fast Loop."""
    station: str = "UNKNOWN"           # WWV, WWVH, CHU
    propagation_mode: str = "UNKNOWN"  # 1F, 2F, 3F, 1E, etc.
    n_hops: int = 1
    propagation_delay_ms: float = 0.0
    snr_db: float = 0.0
    confidence: float = 0.0
    last_update_minute: int = 0
    d_clock_raw_ms: float = 0.0        # Raw D_clock for this channel (for web GUI)
    uncertainty_ms: float = 10.0       # Uncertainty estimate (for web GUI)
    
    def is_valid(self) -> bool:
        """Check if state is valid for Fast Loop use."""
        return (
            self.station != "UNKNOWN" and
            self.propagation_mode != "UNKNOWN" and
            self.propagation_delay_ms > 0
        )


@dataclass
class TimingSolution:
    """Complete timing solution output."""
    timestamp: float                   # Unix timestamp
    rtp_anchor: int                    # RTP timestamp at minute boundary
    utc_ref: float                     # UTC reference (minute boundary)
    d_clock_ms: float                  # System clock offset from UTC
    d_clock_uncertainty_ms: float      # Uncertainty estimate
    clock_drift_ppm: float             # Estimated drift rate
    status: str                        # ACQUIRING, LOCKED, HOLDOVER
    primary_station: str               # Dominant station for timing
    propagation_mode: str              # Current propagation mode
    channels_contributing: int         # Number of channels in fusion
    discrimination: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        """Serialize to JSON for SHM output."""
        return json.dumps(asdict(self), indent=2)


@dataclass
class BroadcastCalibration:
    """
    Per-broadcast calibration offset learned from data.
    
    Each broadcast (station+frequency) has different systematic biases from:
    - Matched filter group delay
    - Frequency-dependent ionospheric delays (1/f²)
    - Detection threshold effects
    
    Calibration brings each broadcast's mean D_clock toward 0 (UTC alignment).
    """
    station: str              # WWV, WWVH, CHU
    frequency_mhz: float      # Broadcast frequency
    offset_ms: float = 0.0    # Calibration offset to apply
    uncertainty_ms: float = 10.0  # Uncertainty in offset
    n_samples: int = 0        # Number of samples used for learning
    last_updated: float = 0.0 # Unix time of last update
    
    @property
    def broadcast_key(self) -> str:
        """Unique key for this broadcast."""
        return f"{self.station}_{self.frequency_mhz:.2f}"


@dataclass
class FusionResult:
    """Result of multi-broadcast D_clock fusion."""
    d_clock_ms: float              # Fused D_clock (calibrated)
    d_clock_raw_ms: float          # Uncalibrated mean
    uncertainty_ms: float          # Weighted std dev
    n_broadcasts: int              # Number contributing
    n_outliers_rejected: int       # Outliers removed
    quality_grade: str             # A/B/C/D
    n_total_candidates: int = 0
    n_prefilter_rejected: int = 0
    
    # Per-station breakdown
    wwv_count: int = 0
    wwvh_count: int = 0
    chu_count: int = 0


@dataclass 
class ChannelBuffer:
    """
    Per-channel ring buffer and odd/even minute buffers.
    
    Ring buffer: Keeps ±1.5s around minute boundary for Fast Loop
    Odd/Even buffers: 60-second buffers for Slow Loop, each spanning -5s to +55s
                      around their respective minute boundaries
    """
    channel_name: str
    ssrc: int
    sample_rate: int = 20000
    
    # Ring buffer for Fast Loop (3 seconds = 60,000 samples)
    ring_size: int = 60000
    ring_buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.complex64))
    ring_write_pos: int = 0
    ring_start_rtp: int = 0
    
    # Odd/Even minute buffers for Slow Loop (60 seconds = 1,200,000 samples each)
    # Each buffer spans from -5s before minute boundary to +55s after
    # This captures the tone at :00 with 5s pre-roll and 55s of content
    minute_buffer_size: int = 1200000  # 60 seconds at 20kHz
    odd_buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.complex64))
    even_buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.complex64))
    odd_write_pos: int = 0
    even_write_pos: int = 0
    odd_start_rtp: int = 0
    even_start_rtp: int = 0
    odd_start_wallclock: float = 0.0
    even_start_wallclock: float = 0.0
    odd_minute: int = -1  # Which odd minute this buffer is for
    even_minute: int = -1  # Which even minute this buffer is for
    
    # RTP tracking
    last_rtp_timestamp: int = 0
    last_wallclock: float = 0.0
    packets_received: int = 0
    
    def __post_init__(self):
        """Initialize buffers."""
        self.ring_buffer = np.zeros(self.ring_size, dtype=np.complex64)
        self.odd_buffer = np.zeros(self.minute_buffer_size, dtype=np.complex64)
        self.even_buffer = np.zeros(self.minute_buffer_size, dtype=np.complex64)
    
    def add_samples(self, samples: np.ndarray, rtp_timestamp: int, wallclock: float):
        """
        Add samples to buffers.
        
        Ring buffer: Circular, always keeps most recent samples
        Odd/Even buffers: Fill based on which minute window we're in
                         Each spans -5s to +65s around its minute boundary
        """
        self.last_rtp_timestamp = rtp_timestamp
        self.last_wallclock = wallclock
        self.packets_received += 1
        
        # FIX (2025-12-12): Gap Stuffing for Timing Integrity
        # If RTP timestamp jumps, we must insert silence to keep buffer indices aligned with time.
        if hasattr(self, 'next_expected_rtp') and self.next_expected_rtp is not None:
            gap = rtp_timestamp - self.next_expected_rtp
            if gap > 0:
                # Sanity check: limit gap stuffing to e.g. 60 seconds to avoid OOM on massive drop
                if gap > 60 * self.sample_rate:
                    logger.warning(f"gap {gap} too large, resetting stream alignment")
                    self.next_expected_rtp = None # Treat as new stream
                else:
                    logger.warning(f"Gap detected: {gap} samples ({gap/self.sample_rate*1000:.1f}ms). Stuffing with zeros.")
                    gap_zeros = np.zeros(gap, dtype=samples.dtype)
                    samples = np.concatenate([gap_zeros, samples])
                    # Shift timestamp back to start of gap
                    rtp_timestamp = self.next_expected_rtp
        
        # Use the full (possibly stuffed) samples for the Slow Loop to maintain minute-alignment
        full_samples = samples
        n_full = len(full_samples)
        
        # Update expected next RTP (for next call)
        self.next_expected_rtp = rtp_timestamp + n_full
        
        # Prepare samples for Ring Buffer (Fast Loop) - Truncate if too large
        if n_full > self.ring_size:
            logger.warning(f"Truncating {n_full - self.ring_size} samples for Ring Buffer (kept full for Slow Loop)")
            ring_samples = full_samples[-self.ring_size:]
            n_ring = self.ring_size
        else:
            ring_samples = full_samples
            n_ring = n_full
        
        # Determine target minute based on wallclock
        current_second = wallclock % 60
        current_minute = int(wallclock) // 60
        
        if current_second >= 55:
            target_minute = current_minute + 1
        else:
            target_minute = current_minute
        
        # Route FULL samples to odd or even buffer (Slow Loop)
        if target_minute % 2 == 1:
            if self.odd_minute != target_minute:
                self.odd_write_pos = 0
                self.odd_start_rtp = 0
                self.odd_start_wallclock = 0.0
                self.odd_minute = target_minute
            self._add_to_odd(full_samples, rtp_timestamp, wallclock, n_full)
        else:
            if self.even_minute != target_minute:
                self.even_write_pos = 0
                self.even_start_rtp = 0
                self.even_start_wallclock = 0.0
                self.even_minute = target_minute
            self._add_to_even(full_samples, rtp_timestamp, wallclock, n_full)
        
        # Ring buffer: circular write using TRUNCATED samples
        write_start = self.ring_write_pos % self.ring_size
        
        if write_start + n_ring <= self.ring_size:
            self.ring_buffer[write_start:write_start + n_ring] = ring_samples
        else:
            first_part = self.ring_size - write_start
            self.ring_buffer[write_start:] = ring_samples[:first_part]
            self.ring_buffer[:n_ring - first_part] = ring_samples[first_part:]
        
        self.ring_write_pos += n_ring
        
        # Update ring_start_rtp correctly using FULL original length
        # Start = End_of_Stream - Ring_Size
        current_end_rtp = rtp_timestamp + n_full
        if self.ring_write_pos >= self.ring_size:
            self.ring_start_rtp = current_end_rtp - self.ring_size
        elif self.packets_received == 1:
            self.ring_start_rtp = rtp_timestamp
    
    def _add_to_odd(self, samples: np.ndarray, rtp_timestamp: int, wallclock: float, n_samples: int):
        """Add samples to odd minute buffer."""
        if self.odd_write_pos + n_samples <= self.minute_buffer_size:
            if self.odd_write_pos == 0:
                self.odd_start_rtp = rtp_timestamp
                self.odd_start_wallclock = wallclock
            self.odd_buffer[self.odd_write_pos:self.odd_write_pos + n_samples] = samples
            self.odd_write_pos += n_samples
    
    def _add_to_even(self, samples: np.ndarray, rtp_timestamp: int, wallclock: float, n_samples: int):
        """Add samples to even minute buffer."""
        if self.even_write_pos + n_samples <= self.minute_buffer_size:
            if self.even_write_pos == 0:
                self.even_start_rtp = rtp_timestamp
                self.even_start_wallclock = wallclock
            self.even_buffer[self.even_write_pos:self.even_write_pos + n_samples] = samples
            self.even_write_pos += n_samples
    
    def get_ring_samples(self) -> tuple[np.ndarray, int]:
        """
        Get ring buffer samples (for Fast Loop).
        
        Returns:
            (samples, start_rtp_timestamp)
        """
        if self.ring_write_pos < self.ring_size:
            # Buffer not full yet
            return self.ring_buffer[:self.ring_write_pos].copy(), self.ring_start_rtp
        else:
            # Return linearized ring buffer
            pos = self.ring_write_pos % self.ring_size
            result = np.concatenate([
                self.ring_buffer[pos:],
                self.ring_buffer[:pos]
            ])
            return result, self.ring_start_rtp
    
    def get_minute_samples(self, minute: int) -> tuple[np.ndarray, int, float]:
        """
        Get samples for a specific minute (for Slow Loop).
        
        Uses odd/even double-buffering: odd minutes use odd_buffer,
        even minutes use even_buffer.
        
        Args:
            minute: Unix minute number to retrieve
            
        Returns:
            (samples, start_rtp_timestamp, start_wallclock)
        """
        if minute % 2 == 1:
            # Odd minute
            if self.odd_minute == minute:
                return self.odd_buffer[:self.odd_write_pos].copy(), self.odd_start_rtp, self.odd_start_wallclock
            else:
                return np.array([], dtype=np.complex64), 0, 0.0
        else:
            # Even minute
            if self.even_minute == minute:
                return self.even_buffer[:self.even_write_pos].copy(), self.even_start_rtp, self.even_start_wallclock
            else:
                return np.array([], dtype=np.complex64), 0, 0.0
    
    def reset_for_new_minute(self):
        """Reset buffers for new minute - NO-OP with double buffering."""
        # With odd/even double buffering, resets happen automatically
        # when the buffer switches to a new minute. This method is kept
        # for API compatibility but does nothing.
        pass


class LiveTimeEngine:
    """
    Live timing engine with Twin-Stream architecture.
    
    Subscribes to RTP multicast and performs real-time timing extraction
    with Fast Loop (tone detection) and Slow Loop (discrimination).
    """
    
    STATE_FILE = "/var/lib/time-manager/state/time_state.json"
    SHM_PATH = "/dev/shm/grape_timing.json"
    
    def __init__(
        self,
        multicast_address: str,
        port: int = 5004,
        channels: List[Dict[str, Any]] = None,
        sample_rate: int = 20000,
        receiver_grid: str = "EM38ww",
        receiver_lat: float = None,
        receiver_lon: float = None,
        enable_chrony: bool = False,
        chrony_unit: int = 0,
        status_address: str = "radiod.local"
    ):
        """
        Initialize live time engine.
        
        Args:
            multicast_address: RTP multicast group
            port: RTP port
            channels: List of channel configs with 'name', 'ssrc', 'frequency_hz'
            sample_rate: Sample rate (Hz)
            receiver_grid: Maidenhead grid square
            receiver_lat: Precise latitude
            receiver_lon: Precise longitude
            enable_chrony: Enable Chrony SHM output
            chrony_unit: Chrony SHM unit number
            status_address: Radiod status multicast for channel discovery
        """
        self.multicast_address = multicast_address
        self.port = port
        self.sample_rate = sample_rate
        self.status_address = status_address
        self.receiver_grid = receiver_grid
        self.receiver_lat = receiver_lat
        self.receiver_lon = receiver_lon
        self.enable_chrony = enable_chrony
        self.chrony_unit = chrony_unit
        
        # Channel configuration
        self.channels = channels or []
        self.channel_buffers: Dict[str, ChannelBuffer] = {}
        self.channel_states: Dict[str, ChannelState] = {}
        
        # Engine state
        self.state = EngineState.STARTING
        self.running = False
        self.current_minute = 0
        self.last_fast_loop_minute = 0
        self.last_slow_loop_minute = 0
        
        # Timing results
        self.d_clock_ms = 0.0
        self.d_clock_uncertainty_ms = 10.0
        self.clock_drift_ppm = 0.0
        
        # RTP-to-System offset smoothing (eliminates host jitter)
        # This is the key to using GPSDO as a "steel ruler"
        self.rtp_system_offset: Dict[str, float] = {}  # Per-channel smoothed offset
        self.rtp_offset_alpha = 0.05  # EMA smoothing factor (slow adaptation)

        # Track RTP continuity per channel (start timestamps in samples)
        self._last_rtp_end: Dict[str, int] = {}

        # Non-finite IQ sample tracking (rate-limit warnings)
        self._nonfinite_log_state: Dict[str, Dict[str, float]] = {}
        
        # RTP receiver (will be initialized on start)
        self.rtp_receiver = None
        
        # Processing threads
        self.fast_loop_thread = None
        self.slow_loop_thread = None
        
        # Chrony SHM
        self.chrony_shm = None
        
        # Timing engines (lazy loaded)
        self._tone_detectors: Dict[str, Any] = {}
        self._discriminators: Dict[str, Any] = {}
        
        # Statistics
        self.stats = {
            'fast_loop_count': 0,
            'slow_loop_count': 0,
            'chrony_updates': 0,
            'start_time': 0.0
        }
        
        # Per-broadcast calibration (station+frequency -> offset)
        self.calibration: Dict[str, BroadcastCalibration] = {}
        self.calibration_file = Path("/var/lib/time-manager/state/broadcast_calibration.json")
        self._load_calibration()
        
        # History for calibration learning (broadcast_key -> list of d_clock values)
        self.calibration_history: Dict[str, deque] = {}
        self.calibration_history_max = 100  # Keep last N measurements per broadcast
        
        # Last fusion result
        self.last_fusion: Optional[FusionResult] = None
        
        # Clock convergence model (Kalman filter for GPSDO systems)
        from ..timing.clock_convergence import ClockConvergenceModel
        self.clock_convergence = ClockConvergenceModel(
            lock_uncertainty_ms=2.0,
            min_samples_for_lock=30,
            anomaly_sigma=3.0,
            max_consecutive_anomalies=5
        )
        
        # Fusion history for web GUI (Kalman funnel chart)
        self.fusion_history: deque = deque(maxlen=1440)  # 24 hours at 1/min
        
        # Chrony history for web GUI (source comparison chart)
        self.chrony_history: deque = deque(maxlen=1440)  # 24 hours at 1/min
        
        # Callback for fusion updates (used by web server)
        self.on_fusion_update: Optional[Callable[[FusionResult], None]] = None
        
        # Rejection monitoring
        self.consecutive_rejections = 0
        
        # TransmissionTimeSolver for computing per-station propagation delays
        self._propagation_solver = None
        if self.receiver_lat is not None and self.receiver_lon is not None:
            try:
                from ..timing.transmission_time_solver import TransmissionTimeSolver
                self._propagation_solver = TransmissionTimeSolver(
                    receiver_lat=self.receiver_lat,
                    receiver_lon=self.receiver_lon,
                    sample_rate=sample_rate
                )
            except Exception as e:
                logger.warning(f"Could not initialize propagation solver: {e}")
        
        logger.info("=" * 60)
        logger.info("LiveTimeEngine initializing")
        logger.info(f"  Multicast: {multicast_address}:{port}")
        logger.info(f"  Channels: {len(self.channels)}")
        logger.info(f"  Sample rate: {sample_rate} Hz")
        logger.info(f"  Chrony: {'enabled' if enable_chrony else 'disabled'}")
        logger.info("=" * 60)
    
    def _init_channel_buffers(self):
        """Initialize per-channel buffers."""
        for ch_config in self.channels:
            name = ch_config['name']
            ssrc = ch_config['ssrc']
            
            self.channel_buffers[name] = ChannelBuffer(
                channel_name=name,
                ssrc=ssrc,
                sample_rate=self.sample_rate
            )
            
            # Initialize state (will be loaded from disk or acquired)
            self.channel_states[name] = ChannelState()
            
            logger.info(f"  Channel buffer: {name} (SSRC={ssrc})")
    
    def _load_state(self) -> bool:
        """
        Load persisted state from disk.
        
        Returns:
            True if state was loaded and is fresh (< 1 hour)
        """
        state_path = Path(self.STATE_FILE)
        
        if not state_path.exists():
            logger.info("No persisted state found, starting fresh")
            return False
        
        try:
            with open(state_path, 'r') as f:
                saved = json.load(f)
            
            # Check age
            save_time = saved.get('timestamp', 0)
            age_seconds = time.time() - save_time
            age_minutes = age_seconds / 60
            
            if age_seconds > 3600:  # 1 hour
                logger.warning(f"State is stale ({age_minutes:.1f} min old), ignoring")
                return False
            
            # Load channel states
            for name, state_dict in saved.get('channels', {}).items():
                if name in self.channel_states:
                    state = self.channel_states[name]
                    state.station = state_dict.get('station', 'UNKNOWN')
                    state.propagation_mode = state_dict.get('propagation_mode', 'UNKNOWN')
                    state.n_hops = state_dict.get('n_hops', 1)
                    state.propagation_delay_ms = state_dict.get('propagation_delay_ms', 0.0)
                    state.snr_db = state_dict.get('snr_db', 0.0)
                    state.confidence = state_dict.get('confidence', 0.0)
            
            logger.info(f"Loaded persisted state ({age_minutes:.1f} min old)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False
    
    def _save_state(self):
        """Save current state to disk."""
        state_path = Path(self.STATE_FILE)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'timestamp': time.time(),
            'd_clock_ms': self.d_clock_ms,
            'd_clock_uncertainty_ms': self.d_clock_uncertainty_ms,
            'clock_drift_ppm': self.clock_drift_ppm,
            'channels': {}
        }
        
        for name, ch_state in self.channel_states.items():
            state['channels'][name] = {
                'station': ch_state.station,
                'propagation_mode': ch_state.propagation_mode,
                'n_hops': ch_state.n_hops,
                'propagation_delay_ms': ch_state.propagation_delay_ms,
                'snr_db': ch_state.snr_db,
                'confidence': ch_state.confidence
            }
        
        # Atomic write
        tmp_path = state_path.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            json.dump(state, f, indent=2)
        tmp_path.rename(state_path)
        
        logger.debug("State saved to disk")
    
    def _load_calibration(self):
        """Load per-broadcast calibration from disk."""
        if self.calibration_file.exists():
            try:
                with open(self.calibration_file, 'r') as f:
                    data = json.load(f)

                max_abs_offset_ms = 500.0
                
                for broadcast_key, cal_data in data.items():
                    # Parse station and frequency from key
                    parts = broadcast_key.rsplit('_', 1)
                    station = parts[0] if len(parts) > 1 else broadcast_key
                    freq = float(parts[1]) if len(parts) > 1 else 0.0

                    offset_ms = float(cal_data.get('offset_ms', 0.0))
                    n_samples = int(cal_data.get('n_samples', 0) or 0)
                    if abs(offset_ms) > max_abs_offset_ms:
                        logger.warning(
                            f"Discarding invalid calibration {broadcast_key}: offset_ms={offset_ms:+.3f}ms"
                        )
                        offset_ms = 0.0
                        n_samples = 0
                    
                    self.calibration[broadcast_key] = BroadcastCalibration(
                        station=station,
                        frequency_mhz=cal_data.get('frequency_mhz', freq),
                        offset_ms=offset_ms,
                        uncertainty_ms=cal_data.get('uncertainty_ms', 10.0),
                        n_samples=n_samples,
                        last_updated=cal_data.get('last_updated', 0.0)
                    )
                
                logger.info(f"Loaded {len(self.calibration)} broadcast calibrations")
            except Exception as e:
                logger.warning(f"Could not load calibration: {e}")
    
    def _save_calibration(self):
        """Save per-broadcast calibration to disk."""
        self.calibration_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {}
        for broadcast_key, cal in self.calibration.items():
            data[broadcast_key] = {
                'station': cal.station,
                'frequency_mhz': cal.frequency_mhz,
                'offset_ms': cal.offset_ms,
                'uncertainty_ms': cal.uncertainty_ms,
                'n_samples': cal.n_samples,
                'last_updated': cal.last_updated
            }
        
        tmp_path = self.calibration_file.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=2)
        tmp_path.rename(self.calibration_file)
    
    def _get_broadcast_key(self, station: str, frequency_mhz: float) -> str:
        """Generate consistent broadcast key for calibration lookups."""
        return f"{station}_{frequency_mhz:.2f}"
    
    def _publish_shm(self, solution: TimingSolution):
        """Publish timing solution to shared memory."""
        shm_path = Path(self.SHM_PATH)
        
        # Atomic write
        tmp_path = shm_path.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            f.write(solution.to_json())
        tmp_path.rename(shm_path)
        
        logger.debug(f"Published to SHM: d_clock={solution.d_clock_ms:.2f}ms")
    
    def _sample_chrony(self):
        """Sample chrony sources and add to history for web GUI."""
        import subprocess
        
        try:
            result = subprocess.run(
                ['chronyc', '-c', 'sources'],
                capture_output=True, text=True, timeout=5
            )
            
            sources = {}
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) >= 10:
                    name = parts[2]
                    offset_s = float(parts[7]) if parts[7] else 0.0
                    sources[name] = {
                        'offset_ms': offset_s * 1000,
                        'state': parts[1],
                        'stratum': int(parts[3]) if parts[3].isdigit() else 0
                    }
            
            # Extract key sources
            tmgr = sources.get('TMGR', {}).get('offset_ms')
            gps = sources.get('192.168.0.134', {}).get('offset_ms')
            
            # Find best NTP (excluding local GPS)
            ntp_sources = {k: v for k, v in sources.items() 
                          if k not in ('TMGR', '192.168.0.134') and not k.startswith('#')}
            best_ntp_name = min(ntp_sources.keys(), 
                               key=lambda k: abs(ntp_sources[k]['offset_ms'])) if ntp_sources else None
            best_ntp = ntp_sources.get(best_ntp_name, {}).get('offset_ms') if best_ntp_name else None
            
            self.chrony_history.append({
                'timestamp': time.time(),
                'tmgr_offset_ms': tmgr,
                'gps_offset_ms': gps,
                'best_ntp_offset_ms': best_ntp,
                'best_ntp_name': best_ntp_name,
                'all_sources': sources
            })
            
        except Exception as e:
            logger.debug(f"Chrony sampling failed: {e}")
    
    def _rtp_callback(self, channel_name: str):
        """
        Create RTP callback for a channel.
        
        Returns a callback function that adds samples to the channel buffer.
        """
        def callback(header, payload: bytes, wallclock: Optional[float]):
            if channel_name not in self.channel_buffers:
                return
            
            # Convert payload to complex64 samples
            samples = np.frombuffer(payload, dtype=np.complex64)

            if not np.isfinite(samples).all():
                bad = int(np.size(samples) - np.count_nonzero(np.isfinite(samples)))
                logger.warning(f"{channel_name}: Non-finite IQ samples in RTP payload (bad={bad}/{len(samples)})")
                samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
            
            # FIX Issue 2: Safe wallclock handling
            # Never estimate absolute time from raw RTP without a known anchor
            if wallclock is None:
                # Use current system time as fallback (introduces jitter but is safe)
                wc = time.time()
            else:
                wc = wallclock
                
                # FIX Issue 1: Update smoothed RTP-to-System offset
                # This filters out USB latency, kernel scheduling, and GC pauses
                rtp_time = header.timestamp / self.sample_rate
                instant_offset = wc - rtp_time
                
                if channel_name in self.rtp_system_offset:
                    # EMA smoothing: slow adaptation preserves GPSDO "steel ruler"
                    old_offset = self.rtp_system_offset[channel_name]
                    self.rtp_system_offset[channel_name] = (
                        (1 - self.rtp_offset_alpha) * old_offset + 
                        self.rtp_offset_alpha * instant_offset
                    )
                else:
                    # Initialize with first observation
                    self.rtp_system_offset[channel_name] = instant_offset

            # Add to buffer
            self.channel_buffers[channel_name].add_samples(
                samples, header.timestamp, wc
            )
            
        return callback

    def _stream_callback(self, channel_name: str, channel_info=None):
        """
        Create callback for RadiodStream.
        
        RadiodStream delivers (samples: np.ndarray, quality: StreamQuality).
        We adapt this to add samples to our channel buffer.
        
        Critical: We must use actual RTP timestamps from radiod, not sample counts.
        RTP timestamps are GPS-disciplined and essential for sub-ms timing.
        
        Args:
            channel_name: Name of the channel
            channel_info: ChannelInfo (not used - gps_time/rtp_timesnap become stale)
        """
        def callback(samples: np.ndarray, quality):
            if channel_name not in self.channel_buffers:
                return

            if not np.isfinite(samples).all():
                bad = int(np.size(samples) - np.count_nonzero(np.isfinite(samples)))
                bad_frac = (bad / float(np.size(samples))) if np.size(samples) else 0.0
                now = time.time()
                state = self._nonfinite_log_state.get(channel_name)
                if state is None:
                    state = {'last_log': 0.0, 'suppressed': 0.0, 'diag_logged': 0.0}
                    self._nonfinite_log_state[channel_name] = state

                # One-time diagnostics per channel to help identify upstream encoding/preset issues
                if float(state.get('diag_logged', 0.0)) == 0.0:
                    try:
                        finite = samples[np.isfinite(samples)]
                        abs_finite = np.abs(finite) if finite.size else np.array([], dtype=np.float64)
                        abs_min = float(np.min(abs_finite)) if abs_finite.size else 0.0
                        abs_max = float(np.max(abs_finite)) if abs_finite.size else 0.0
                    except Exception:
                        abs_min, abs_max = 0.0, 0.0

                    logger.warning(
                        f"{channel_name}: Non-finite IQ diagnostics: dtype={getattr(samples, 'dtype', None)}, "
                        f"shape={getattr(samples, 'shape', None)}, bad={bad}/{np.size(samples)} ({bad_frac*100:.1f}%), "
                        f"|x| range≈[{abs_min:.3g}, {abs_max:.3g}]"
                    )
                    state['diag_logged'] = 1.0

                # Rate-limit to at most once per 5 seconds per channel
                if now - float(state['last_log']) >= 5.0:
                    suppressed = int(state.get('suppressed', 0.0))
                    extra = f" (+{suppressed} suppressed)" if suppressed > 0 else ""
                    logger.warning(
                        f"{channel_name}: Non-finite IQ samples in stream batch "
                        f"(bad={bad}/{len(samples)}={bad_frac*100:.1f}%){extra}"
                    )
                    state['last_log'] = now
                    state['suppressed'] = 0.0
                else:
                    state['suppressed'] = float(state.get('suppressed', 0.0)) + 1.0

                # Treat corruption as a "gap" rather than mixing partially-corrupt samples.
                # This preserves downstream DSP assumptions and matches the resequencer philosophy.
                if bad_frac >= 0.05:
                    samples = np.zeros_like(samples)
                else:
                    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)

            # Calculate RTP timestamp for this batch from StreamQuality
            if hasattr(quality, 'first_rtp_timestamp') and hasattr(quality, 'batch_start_sample'):
                rtp_timestamp = quality.first_rtp_timestamp + quality.batch_start_sample
            elif hasattr(quality, 'last_rtp_timestamp'):
                rtp_timestamp = quality.last_rtp_timestamp - len(samples)
            else:
                rtp_timestamp = getattr(quality, 'total_samples_delivered', 0)
                logger.warning(f"{channel_name}: No RTP timestamp in StreamQuality, using sample count")

            # Detect discontinuities: ka9q resequencer can skip ahead on packet loss.
            # Any significant gap will corrupt minute buffers and produce large D_clock errors.
            prev_end = self._last_rtp_end.get(channel_name)
            if prev_end is not None:
                gap_samples = int(rtp_timestamp - prev_end)
                # Match grape-recorder resequencer behavior:
                # - Fill smaller forward gaps with zeros to preserve sample-count integrity
                # - Treat very large jumps as discontinuities (radiod restart / channel recreation)
                DISCONTINUITY_THRESHOLD_SAMPLES = 200_000  # 10 seconds at 20 kHz
                MAX_GAP_FILL_SAMPLES = 200_000             # Fill up to 10 seconds with zeros

                if abs(gap_samples) > DISCONTINUITY_THRESHOLD_SAMPLES:
                    gap_ms = (gap_samples / self.sample_rate) * 1000.0
                    logger.warning(
                        f"{channel_name}: RTP discontinuity detected (gap={gap_samples} samples, {gap_ms:+.1f}ms) "
                        f"- resetting minute buffers"
                    )
                    buf = self.channel_buffers.get(channel_name)
                    if buf is not None:
                        buf.reset_for_new_minute()

                    # Drop this batch: it is not contiguous with the previous data.
                    self._last_rtp_end.pop(channel_name, None)
                    return

                if gap_samples > 0:
                    # Forward gap: fill with zeros to keep RTP sample timeline continuous.
                    if gap_samples <= MAX_GAP_FILL_SAMPLES:
                        gap_ms = (gap_samples / self.sample_rate) * 1000.0
                        logger.warning(
                            f"{channel_name}: RTP gap fill (gap={gap_samples} samples, {gap_ms:.1f}ms)"
                        )
                        zeros = np.zeros(gap_samples, dtype=np.complex64)
                        samples = np.concatenate([zeros, samples])
                        rtp_timestamp = int(prev_end)
                    else:
                        gap_ms = (gap_samples / self.sample_rate) * 1000.0
                        logger.warning(
                            f"{channel_name}: RTP gap too large to fill (gap={gap_samples} samples, {gap_ms:.1f}ms) "
                            f"- resetting minute buffers"
                        )
                        buf = self.channel_buffers.get(channel_name)
                        if buf is not None:
                            buf.reset_for_new_minute()
                        self._last_rtp_end.pop(channel_name, None)
                        return
                elif gap_samples < 0:
                    # Backward jump (out-of-order or restart): treat as discontinuity.
                    gap_ms = (gap_samples / self.sample_rate) * 1000.0
                    logger.warning(
                        f"{channel_name}: RTP backward jump detected (gap={gap_samples} samples, {gap_ms:+.1f}ms) "
                        f"- resetting minute buffers"
                    )
                    buf = self.channel_buffers.get(channel_name)
                    if buf is not None:
                        buf.reset_for_new_minute()
                    self._last_rtp_end.pop(channel_name, None)
                    return

            # Maintain smoothed RTP->System offset in RadiodStream mode.
            # We don't get an explicit wallclock from the stream, so use local wallclock
            # and rely on slow EMA smoothing to reject host jitter.
            wc = None
            for attr in ('wallclock', 'wallclock_time', 'system_time', 'timestamp'):
                if hasattr(quality, attr):
                    try:
                        wc = float(getattr(quality, attr))
                        break
                    except Exception:
                        wc = None
            if wc is None:
                wc = time.time()

            rtp_time = rtp_timestamp / self.sample_rate
            instant_offset = wc - rtp_time
            if channel_name in self.rtp_system_offset:
                old_offset = self.rtp_system_offset[channel_name]
                self.rtp_system_offset[channel_name] = (
                    (1 - self.rtp_offset_alpha) * old_offset +
                    self.rtp_offset_alpha * instant_offset
                )
            else:
                self.rtp_system_offset[channel_name] = instant_offset
                logger.info(f"{channel_name}: Initialized RTP→wallclock offset = {instant_offset:.6f}s")

            # Feed the buffer a consistent wallclock derived from RTP sequencing.
            # ChannelBuffer uses wallclock%60 to decide which minute buffer to fill.
            # Using time.time() here reintroduces per-channel scheduling jitter and
            # can cause buffers to be "too early" or "too late" by tens of seconds.
            wallclock_for_buffer = (rtp_timestamp / self.sample_rate) + self.rtp_system_offset.get(channel_name, instant_offset)

            # Add to buffer - odd/even double-buffering handles resets automatically
            self.channel_buffers[channel_name].add_samples(samples, rtp_timestamp, wallclock_for_buffer)

            # Update continuity tracker
            self._last_rtp_end[channel_name] = int(rtp_timestamp + len(samples))

        return callback
    
    def _discover_channels(self, status_address: str = "radiod.local") -> List[Dict[str, Any]]:
        """
        Discover available channels from radiod using ka9q.
        
        Args:
            status_address: Multicast address for radiod status (default: radiod.local)
        
        Returns:
            List of channel configs with name, ssrc, frequency_hz
        """
        try:
            from ka9q import discover_channels
            
            logger.info(f"  Discovering channels from {status_address}...")
            discovered = discover_channels(status_address, listen_duration=3.0)
            channels = []
            
            # discover_channels returns Dict[ssrc, ChannelInfo]
            logger.info(f"  Found {len(discovered)} channels")
            
            # Map frequencies to channel names
            # Known time signal frequencies
            FREQ_TO_STATION = {
                2500000: "WWV 2.5 MHz",
                3330000: "CHU 3.33 MHz",
                5000000: "WWV 5 MHz",
                7850000: "CHU 7.85 MHz",
                10000000: "WWV 10 MHz",
                14670000: "CHU 14.67 MHz",
                15000000: "WWV 15 MHz",
                20000000: "WWV 20 MHz",
                25000000: "WWV 25 MHz",
            }
            
            for ssrc_key, ch_info in discovered.items():
                freq = getattr(ch_info, 'frequency', 0)
                ssrc = ssrc_key  # SSRC is the dict key
                
                # Look up channel name from frequency
                # Round to nearest kHz for matching
                freq_rounded = round(freq / 1000) * 1000
                name = FREQ_TO_STATION.get(freq_rounded)
                
                if not name:
                    # Not a known time signal frequency
                    logger.debug(f"  Skipping unknown frequency: {freq/1e6:.3f} MHz")
                    continue
                
                channels.append({
                    'name': name,
                    'ssrc': ssrc,
                    'frequency_hz': freq,
                    'channel_info': ch_info  # Keep for timing
                })
                logger.info(f"  Discovered: {name} (SSRC={ssrc})")
            
            return channels
            
        except Exception as e:
            logger.error(f"Channel discovery failed: {e}")
            return []
    
    def _init_rtp_receiver(self):
        """
        Initialize RTP streams using ka9q-python directly.
        
        Uses RadiodStream for each channel - handles multicast, resequencing, gaps.
        Uses RadiodControl to create/configure channels if needed.
        """
        from ka9q import (
            discover_channels, RadiodStream, RadiodControl, allocate_ssrc, Encoding
        )
        import hashlib
        
        # Define our time signal frequencies
        TIME_SIGNAL_FREQS = [
            {'frequency_hz': 2500000, 'description': 'WWV 2.5 MHz'},
            {'frequency_hz': 3330000, 'description': 'CHU 3.33 MHz'},
            {'frequency_hz': 5000000, 'description': 'WWV 5 MHz'},
            {'frequency_hz': 7850000, 'description': 'CHU 7.85 MHz'},
            {'frequency_hz': 10000000, 'description': 'WWV 10 MHz'},
            {'frequency_hz': 14670000, 'description': 'CHU 14.67 MHz'},
            {'frequency_hz': 15000000, 'description': 'WWV 15 MHz'},
            {'frequency_hz': 20000000, 'description': 'WWV 20 MHz'},
            {'frequency_hz': 25000000, 'description': 'WWV 25 MHz'},
        ]
        
        CHANNEL_DEFAULTS = {
            'preset': 'iq',
            'sample_rate': 20000,
            'agc': 0,
            'gain': 0.0,
        }

        # Identity/signature fields (SSRC is treated as an internal handle only)
        desired_preset = CHANNEL_DEFAULTS['preset']
        desired_sample_rate = int(CHANNEL_DEFAULTS['sample_rate'])
        desired_encoding = Encoding.F32
        
        # Generate our destination multicast (deterministic from station/instrument ID)
        def generate_multicast_ip(station_id: str, instrument_id: str) -> str:
            combined = f"{station_id}:{instrument_id}"
            h = hashlib.md5(combined.encode()).digest()
            return f"239.{h[0]}.{h[1]}.{h[2]}"
        
        our_destination = generate_multicast_ip('S000171', '173')  # time-manager
        logger.info(f"  time-manager destination: {our_destination}")
        
        # Discover existing channels
        logger.info(f"  Discovering channels from {self.status_address}...")
        discovered = discover_channels(self.status_address, listen_duration=2.0)
        
        # Build signature -> ChannelInfo lookup (SSRC is not identity)
        # Signature = (frequency_hz, sample_rate, preset, destination)
        sig_to_channel = {}
        for _ssrc, ch_info in discovered.items():
            try:
                freq_hz = int(round(float(getattr(ch_info, 'frequency'))))
            except Exception:
                continue
            sr = getattr(ch_info, 'sample_rate', None)
            preset = getattr(ch_info, 'preset', None)
            dest = getattr(ch_info, 'multicast_address', None)
            sig = (freq_hz, sr, preset, dest)
            sig_to_channel[sig] = ch_info

        def _ensure_f32_encoding(ssrc: int, chan_label: str) -> bool:
            try:
                import time as time_mod
                for delay_s in (0.0, 0.2, 0.5, 1.0):
                    if delay_s:
                        time_mod.sleep(delay_s)
                    status = control.tune(ssrc=ssrc, encoding=int(desired_encoding), timeout=3.0)
                    enc = status.get('encoding')
                    if enc == int(desired_encoding):
                        return True
                logger.warning(
                    f"  {chan_label}: Channel encoding is {enc} (expected {int(desired_encoding)}=F32)"
                )
                return False
            except Exception as e:
                logger.warning(f"  {chan_label}: Failed to set/verify encoding=F32 via tune(): {e}")
                return False
        
        logger.info(f"  Found {len(discovered)} existing channels")
        
        # Get RadiodControl for creating/configuring channels
        control = RadiodControl(self.status_address)
        
        # Ensure each time signal frequency has a channel
        self.channels = []
        self._streams = []  # RadiodStream instances
        
        for ch_spec in TIME_SIGNAL_FREQS:
            freq_hz = int(ch_spec['frequency_hz'])
            name = ch_spec['description']

            # Look for an existing channel that matches our desired signature.
            # Destination is part of the identity; if destination differs, treat as absent.
            candidate = None
            for sig, ch_info in sig_to_channel.items():
                (f, sr, preset, dest) = sig
                if f != freq_hz:
                    continue
                if sr is not None and int(sr) != desired_sample_rate:
                    continue
                if preset is not None and str(preset) != str(desired_preset):
                    continue
                if dest != our_destination:
                    continue
                candidate = ch_info
                break

            if candidate is not None:
                ch_info = candidate
                logger.info(f"  ✓ {name}: SSRC={ch_info.ssrc}")

                # Ensure payload encoding is float32 (F32). Do NOT accept S16LE.
                _ensure_f32_encoding(ch_info.ssrc, name)
            else:
                # If a channel already exists on OUR destination with the right frequency,
                # we may reconfigure it in-place. This avoids hijacking channels used by
                # other clients, because destination is unique per app.
                our_dest_candidate = None
                for sig, ch_info_existing in sig_to_channel.items():
                    (f, _sr, _preset, dest) = sig
                    if f == freq_hz and dest == our_destination:
                        our_dest_candidate = ch_info_existing
                        break

                if our_dest_candidate is not None:
                    ch_info = our_dest_candidate
                    logger.info(f"  ⚙ {name}: reconfiguring existing channel on our destination (SSRC={ch_info.ssrc})")
                    try:
                        reconfig_kwargs = {
                            'frequency_hz': freq_hz,
                            'preset': desired_preset,
                            'sample_rate': desired_sample_rate,
                            'destination': our_destination,
                            'ssrc': ch_info.ssrc,
                        }
                        control.create_channel(**reconfig_kwargs)

                        import time as time_mod
                        discovered = {}
                        for delay_s in (0.3, 0.5, 0.8, 1.2, 1.7):
                            time_mod.sleep(delay_s)
                            discovered = discover_channels(self.status_address, listen_duration=1.0)
                            refreshed = discovered.get(ch_info.ssrc)
                            if refreshed is not None:
                                ch_info = refreshed
                                break
                        logger.info(f"  ✓ {name}: SSRC={ch_info.ssrc}")

                        # Ensure payload encoding is float32 (F32).
                        _ensure_f32_encoding(ch_info.ssrc, name)
                    except Exception as e:
                        logger.warning(f"  Failed to reconfigure {name} on our destination: {e}")

                else:
                    # Request channel creation matching our signature.
                    # Prefer radiod-managed SSRC allocation; fall back if API requires explicit SSRC.
                    logger.info(f"  ➕ {name}: creating...")
                    try:
                        create_kwargs = {
                            'frequency_hz': freq_hz,
                            'preset': desired_preset,
                            'sample_rate': desired_sample_rate,
                            'destination': our_destination,
                        }

                        try:
                            control.create_channel(**create_kwargs)
                        except TypeError:
                            # Older API requires explicit SSRC
                            try:
                                ssrc = allocate_ssrc(frequency_hz=freq_hz)
                            except TypeError:
                                try:
                                    ssrc = allocate_ssrc(freq_hz)
                                except TypeError:
                                    ssrc = allocate_ssrc()
                            control.create_channel(ssrc=ssrc, **create_kwargs)

                        import time as time_mod
                        discovered = {}
                        for delay_s in (0.3, 0.5, 0.8, 1.2, 1.7):
                            time_mod.sleep(delay_s)
                            discovered = discover_channels(self.status_address, listen_duration=1.0)
                            if discovered:
                                break

                        # Refresh signature map and locate the created channel by signature.
                        sig_to_channel = {}
                        for _ssrc, _ch_info in discovered.items():
                            try:
                                _freq_hz = int(round(float(getattr(_ch_info, 'frequency'))))
                            except Exception:
                                continue
                            _sr = getattr(_ch_info, 'sample_rate', None)
                            _preset = getattr(_ch_info, 'preset', None)
                            _dest = getattr(_ch_info, 'multicast_address', None)
                            sig_to_channel[(_freq_hz, _sr, _preset, _dest)] = _ch_info

                        for sig, _ch_info in sig_to_channel.items():
                            (f, sr, preset, dest) = sig
                            if f == freq_hz and dest == our_destination:
                                if sr is not None and int(sr) != desired_sample_rate:
                                    continue
                                if preset is not None and str(preset) != str(desired_preset):
                                    continue
                                ch_info = _ch_info
                                break
                        else:
                            logger.warning(f"    Channel creation unverified")
                            continue

                        # Ensure payload encoding is float32 (F32) after creation.
                        _ensure_f32_encoding(ch_info.ssrc, name)
                    except Exception as e:
                        logger.error(f"    Failed to create {name}: {e}")
                        continue
            
            self.channels.append({
                'name': name,
                'ssrc': ch_info.ssrc,
                'frequency_hz': freq_hz,
                'channel_info': ch_info
            })
        
        if not self.channels:
            raise RuntimeError("No channels available")
        
        logger.info(f"  {len(self.channels)} channels ready")
        
        # Re-initialize buffers for these channels
        self._init_channel_buffers()
        
        # Create RadiodStream for each channel
        for ch_config in self.channels:
            name = ch_config['name']
            ch_info = ch_config['channel_info']
            
            # Create callback that adds samples to our buffer
            # Pass channel_info for GPS-accurate wallclock conversion
            callback = self._stream_callback(name, channel_info=ch_info)
            
            stream = RadiodStream(
                channel=ch_info,
                on_samples=callback,
                samples_per_packet=400,  # 20ms at 20kHz
                deliver_interval_packets=1  # Deliver every packet
            )
            stream.start()
            self._streams.append(stream)
            logger.info(f"  Started stream: {name} (SSRC={ch_info.ssrc})")
    
    def _fast_loop(self):
        """
        Fast Loop thread - runs at T=:01 each minute.
        
        Uses ring buffer (±1.5s around minute boundary) and previous
        minute's state to compute D_clock with ~1 second latency.
        """
        logger.info("Fast Loop thread started")
        
        while self.running:
            try:
                # Wait for second :01 of the minute
                now = time.time()
                current_second = now % 60
                
                # Calculate time to :01
                if current_second < 1:
                    wait_time = 1 - current_second
                else:
                    wait_time = 61 - current_second
                
                # Sleep with periodic checks for shutdown
                while wait_time > 0 and self.running:
                    sleep_time = min(wait_time, 0.5)
                    time.sleep(sleep_time)
                    wait_time -= sleep_time
                
                if not self.running:
                    break
                
                # Check current minute
                now = time.time()
                current_minute = int(now / 60)
                
                if current_minute <= self.last_fast_loop_minute:
                    continue  # Already processed this minute
                
                self.last_fast_loop_minute = current_minute
                self.stats['fast_loop_count'] += 1
                
                # Process Fast Loop
                self._process_fast_loop(current_minute)
                
            except Exception as e:
                logger.exception(f"Fast Loop error: {e}")
                time.sleep(1)
        
        logger.info("Fast Loop thread stopped")
    
    def _process_fast_loop(self, minute: int):
        """
        Process Fast Loop for a minute.
        
        Args:
            minute: Unix minute number
        """
        logger.info(f"Fast Loop: minute {minute}")
        
        # Log buffer stats
        total_samples = sum(b.ring_write_pos for b in self.channel_buffers.values())
        total_packets = sum(b.packets_received for b in self.channel_buffers.values())
        logger.info(f"  Buffer stats: {total_samples} samples, {total_packets} packets across {len(self.channel_buffers)} channels")
        
        # Skip if still acquiring
        if self.state == EngineState.ACQUIRING:
            logger.info("  Skipping (still acquiring)")
            return
        
        d_clock_values = []
        minute_boundary = minute * 60  # UTC seconds
        
        for name, buffer in self.channel_buffers.items():
            state = self.channel_states.get(name)
            
            if not state or not state.is_valid():
                logger.debug(f"  {name}: No valid state")
                continue
            
            # Get ring buffer samples (3 seconds around minute boundary)
            samples, start_rtp = buffer.get_ring_samples()
            
            if len(samples) < self.sample_rate * 2:  # Need at least 2 seconds
                logger.debug(f"  {name}: Insufficient samples ({len(samples)})")
                continue
            
            try:
                # Get or create tone detector for this channel
                if name not in self._tone_detectors:
                    from ..timing import MultiStationToneDetector
                    # Use 20kHz sample rate (full rate, not decimated)
                    self._tone_detectors[name] = MultiStationToneDetector(name, self.sample_rate)
                
                detector = self._tone_detectors[name]
                
                # Calculate buffer midpoint time for tone detector
                # The tone detector expects timestamp to be the buffer MIDPOINT
                # 
                # Use the wallclock of the last packet as the buffer end time,
                # then subtract half the buffer duration to get the midpoint.
                # This ties the detection window to GPS-synced system time.
                buffer_duration = len(samples) / self.sample_rate
                buffer_end_time = buffer.last_wallclock
                buffer_midpoint_time = buffer_end_time - (buffer_duration / 2)
                
                # Run tone detection with narrow window (Fast Loop uses previous state)
                # Tones are at :00 (start of NEXT minute).
                # We are at T=:01 of current minute.
                # The tone we just heard is for the CURRENT minute boundary (second 0).
                current_minute_boundary = float(minute * 60)
                
                detections = detector.process_samples(
                    timestamp=buffer_midpoint_time,
                    samples=samples,
                    rtp_timestamp=start_rtp,
                    original_sample_rate=self.sample_rate,
                    search_window_ms=300.0,  # Trusted clock constraint
                    expected_offset_ms=self.clock_convergence.d_clock_ms,
                    minute_boundary_timestamp=current_minute_boundary
                )
                
                if detections:
                    # FIX Issue 3: Filter detections to match channel's current state
                    # Prevents "split brain" where we detect Hawaii but subtract Colorado delay
                    matching_detection = None
                    for d in detections:
                        detection_station = d.station.value if hasattr(d.station, 'value') else str(d.station)
                        if detection_station.upper() == state.station.upper():
                            if matching_detection is None or (d.snr_db or 0) > (matching_detection.snr_db or 0):
                                matching_detection = d
                    
                    if matching_detection:
                        # D_clock = T_arrival - T_propagation
                        # timing_error_ms is offset from expected minute boundary
                        d_clock = matching_detection.timing_error_ms - state.propagation_delay_ms
                        d_clock_values.append((d_clock, state.confidence, matching_detection.snr_db or 1.0))
                        logger.info(f"  {name}: D_clock={d_clock:+.2f}ms, SNR={matching_detection.snr_db:.1f}dB")
                    else:
                        # Station mismatch - don't mix delays!
                        detected_stations = [d.station.value if hasattr(d.station, 'value') else str(d.station) for d in detections]
                        logger.warning(f"  {name}: Station mismatch - expected {state.station}, detected {detected_stations}")
                else:
                    logger.debug(f"  {name}: No tone detected")
                    
            except Exception as e:
                logger.warning(f"  {name}: Tone detection failed: {e}")
        
        if d_clock_values:
            # Weighted average (weight by confidence * SNR)
            total_weight = sum(conf * snr for _, conf, snr in d_clock_values)
            if total_weight > 0:
                self.d_clock_ms = sum(d * conf * snr for d, conf, snr in d_clock_values) / total_weight
                self.d_clock_uncertainty_ms = 5.0 / len(d_clock_values)  # Rough estimate
            
            # Update Chrony if enabled and have good data
            if self.enable_chrony and self.chrony_shm and len(d_clock_values) >= 2:
                self._update_chrony()
                self.stats['chrony_updates'] += 1
            
            logger.info(f"  Fast Loop result: D_clock={self.d_clock_ms:+.2f}ms from {len(d_clock_values)} channels")
        else:
            logger.warning(f"  Fast Loop: No valid detections")
    
    def _slow_loop(self):
        """
        Slow Loop thread - runs at T=:06 each minute.
        
        Uses full minute buffer for complete discrimination and
        state update for next Fast Loop iteration.
        """
        logger.info("Slow Loop thread started")
        
        while self.running:
            try:
                # Wait for second :06 of the minute (after buffer completes at :05)
                now = time.time()
                current_second = now % 60
                
                # Calculate time to :06
                if current_second < 6:
                    wait_time = 6 - current_second
                else:
                    wait_time = 66 - current_second
                
                # Sleep with periodic checks
                while wait_time > 0 and self.running:
                    sleep_time = min(wait_time, 0.5)
                    time.sleep(sleep_time)
                    wait_time -= sleep_time
                
                if not self.running:
                    break
                
                # Check current minute
                now = time.time()
                current_minute = int(now / 60)
                
                if current_minute <= self.last_slow_loop_minute:
                    continue
                
                self.last_slow_loop_minute = current_minute
                self.stats['slow_loop_count'] += 1
                
                # Process Slow Loop
                self._process_slow_loop(current_minute - 1)  # Process previous minute
                
                # Sample chrony sources for history
                self._sample_chrony()
                
            except Exception as e:
                logger.exception(f"Slow Loop error: {e}")
                time.sleep(1)
        
        logger.info("Slow Loop thread stopped")
    
    def _process_slow_loop(self, minute: int):
        """
        Process Slow Loop for a minute.
        
        Args:
            minute: Unix minute number (previous minute)
        """
        logger.info(f"Slow Loop: minute {minute}")
        
        results = {}
        
        # Get or create Phase2 engines for full analysis
        if not hasattr(self, '_phase2_engines'):
            self._phase2_engines = {}
        
        for name, buffer in self.channel_buffers.items():
            # Get samples for the requested minute using odd/even double-buffering
            samples, start_rtp, start_wallclock = buffer.get_minute_samples(minute)

            if len(samples) > 0 and not np.isfinite(samples).all():
                bad = int(np.size(samples) - np.count_nonzero(np.isfinite(samples)))
                logger.warning(f"  {name}: Non-finite IQ samples in minute buffer (bad={bad}/{len(samples)}) - sanitizing")
                samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)

            # IMPORTANT: Prefer RTP-based wallclock if available.
            # `start_wallclock` is the arrival time of the packet, which includes
            # network jitter and buffering latency (typically ~25ms).
            # `rtp_based_start_wallclock` uses the Smoothed Round Trip offset
            # (or just average offset) which filters out jitter/lag.
            rtp_based_start_wallclock = None
            if start_rtp and start_rtp > 0:
                rtp_offset = self.rtp_system_offset.get(name)
                if rtp_offset is not None:
                    # RTP timestamp -> Wallclock using averaged offset
                    rtp_based_start_wallclock = (start_rtp / self.sample_rate) + rtp_offset
            
            # Decide which timestamp to use
            if rtp_based_start_wallclock is not None:
                effective_start_wallclock = rtp_based_start_wallclock
                # logger.debug(f"  {name}: Using RTP-based wallclock (arrival_lag={start_wallclock - effective_start_wallclock:.3f}s)")
            elif start_wallclock and start_wallclock > 0:
                effective_start_wallclock = start_wallclock
                logger.debug(f"  {name}: Using packet arrival wallclock (no RTP offset map)")
            else:
                effective_start_wallclock = None
            
            # Debug: show buffer state
            buffer_duration_s = len(samples) / self.sample_rate if len(samples) > 0 else 0
            if minute % 2 == 1:
                logger.info(f"  {name}: odd_minute={buffer.odd_minute}, requested={minute}, samples={len(samples)} ({buffer_duration_s:.1f}s), effective_start={effective_start_wallclock or 0.0:.3f}")
            else:
                logger.info(f"  {name}: even_minute={buffer.even_minute}, requested={minute}, samples={len(samples)} ({buffer_duration_s:.1f}s), effective_start={effective_start_wallclock or 0.0:.3f}")
            
            min_needed = int(self.sample_rate * 55)  # Need most of the minute window
            if len(samples) < min_needed:
                logger.info(f"  {name}: Insufficient samples ({len(samples)}) for minute {minute} (need {min_needed})")
                buffer.reset_for_new_minute()
                continue
            
            # FIX 5 (2025-12-11): Verify buffer contains the minute boundary
            # The tone is at second 0 of the target minute. The buffer should start
            # at :55 of the previous minute (5 seconds before the tone).
            # If start_wallclock is AFTER the minute boundary, we missed the tone.
            minute_boundary_float = float(minute * 60)
            if effective_start_wallclock > minute_boundary_float:
                seconds_late = effective_start_wallclock - minute_boundary_float
                logger.warning(f"  {name}: Buffer started {seconds_late:.1f}s AFTER minute boundary - skipping (missed tone)")
                continue
            
            # Also check if buffer is too early (started before :55 of prev minute)
            expected_start = minute_boundary_float - 5.0
            if effective_start_wallclock < expected_start - 1.0:  # Allow 1s tolerance
                seconds_early = expected_start - effective_start_wallclock
                logger.warning(f"  {name}: Buffer started {seconds_early:.1f}s too early - may have stale data")
            
            try:
                # Get or create Phase2 engine for this channel
                if name not in self._phase2_engines:
                    from ..timing import Phase2TemporalEngine
                    from pathlib import Path
                    
                    # Extract frequency from channel name
                    freq_hz = next(
                        (ch['frequency_hz'] for ch in self.channels if ch['name'] == name),
                        10000000  # Default 10 MHz
                    )
                    
                    self._phase2_engines[name] = Phase2TemporalEngine(
                        raw_archive_dir=Path("/tmp"),  # Not used for RAM processing
                        output_dir=Path("/tmp"),       # Not used
                        channel_name=name,
                        frequency_hz=freq_hz,
                        receiver_grid=self.receiver_grid,
                        sample_rate=self.sample_rate,
                        precise_lat=self.receiver_lat,
                        precise_lon=self.receiver_lon
                    )
                
                engine = self._phase2_engines[name]
                
                # FIX 1 (2025-12-11): Use ACTUAL buffer start time, not calculated
                # 
                # Previously: system_time = float(minute * 60) - 5.0
                # This assumed the buffer always starts exactly at :55 of the previous
                # minute. But if there's any latency or the buffer doesn't start exactly
                # at :55, this creates a timing offset that corrupts D_clock.
                #
                # Now: Use start_wallclock from the buffer, which is the actual system
                # time when the first sample arrived. This ties the timing directly to
                # the GPS-disciplined system clock.
                #
                # If start_wallclock is 0 (not set), fall back to calculated value.
                if effective_start_wallclock and effective_start_wallclock > 0:
                    system_time = effective_start_wallclock
                else:
                    # Fallback to calculated (should rarely happen)
                    system_time = float(minute * 60) - 5.0
                    logger.warning(f"  {name}: start_wallclock not set, using calculated system_time={system_time:.3f}")
                
                # Adaptive search window:
                # If locked, narrow the search to expected location to reject noise.
                # If acquiring, use wide window.
                if self.clock_convergence.is_locked and self.clock_convergence.d_clock_ms is not None:
                    current_search_window_ms = 200.0 # Narrow window when locked
                    current_expected_offset_ms = self.clock_convergence.d_clock_ms
                else:
                    current_search_window_ms = 1500.0 # Wide window for acquisition
                    current_expected_offset_ms = None
                
                # Run full Phase 2 analysis
                phase2_result = engine.process_minute(
                    iq_samples=samples,
                    system_time=system_time,
                    rtp_timestamp=start_rtp,
                    minute_boundary=minute_boundary_float, # Pass minute boundary explicitly
                    search_window_ms=current_search_window_ms,
                    expected_offset_ms=current_expected_offset_ms
                )
                
                if phase2_result and phase2_result.d_clock_ms is not None:
                    # Extract from nested result objects
                    solution = phase2_result.solution
                    time_snap = phase2_result.time_snap
                    
                    # Extract ALL detected broadcasts on this frequency
                    # (not just the dominant station)
                    broadcasts_on_freq = self._extract_all_broadcasts(
                        name, phase2_result, solution, time_snap
                    )
                    
                    for broadcast_key, broadcast_result in broadcasts_on_freq.items():
                        results[broadcast_key] = broadcast_result
                        logger.info(f"  {broadcast_key}: {broadcast_result['station']} "
                                   f"{broadcast_result['propagation_mode']} "
                                   f"D_clock={broadcast_result['d_clock_ms']:+.2f}ms "
                                   f"SNR={broadcast_result['snr_db']:.1f}dB")
                    
                    # Update channel state for next Fast Loop (use primary/dominant)
                    state = self.channel_states[name]
                    primary = broadcasts_on_freq.get(name, list(broadcasts_on_freq.values())[0])
                    state.station = primary['station']
                    state.propagation_mode = primary['propagation_mode']
                    state.propagation_delay_ms = primary['propagation_delay_ms']
                    state.snr_db = primary['snr_db']
                    state.confidence = primary['confidence']
                    state.last_update_minute = minute
                else:
                    logger.warning(f"  {name}: Phase 2 returned no result")
                    
            except Exception as e:
                logger.warning(f"  {name}: Phase 2 processing failed: {e}")
            
            # Don't reset the full buffer here - it needs to keep accumulating
            # so it contains the minute boundary. The buffer will naturally
            # overflow and stop accepting new samples, which is fine since
            # we only need ~60 seconds of data.
        
        # Transition from ACQUIRING to TRACKING
        if self.state == EngineState.ACQUIRING and results:
            self.state = EngineState.TRACKING
            logger.info("State transition: ACQUIRING -> TRACKING")
        
        applied_clock_update = False

        # Fuse D_clock from ALL broadcasts (up to 13) using improved algorithm
        if results:
            fusion_result = self._fuse_broadcasts(results)
            
            if fusion_result:
                logger.info(
                    f"  Fusion summary: candidates={fusion_result.n_total_candidates} "
                    f"prefilter_rejected={fusion_result.n_prefilter_rejected} "
                    f"mad_rejected={fusion_result.n_outliers_rejected} "
                    f"used={fusion_result.n_broadcasts} grade={fusion_result.quality_grade} "
                    f"uncertainty={fusion_result.uncertainty_ms:.3f}ms"
                )

                allow_clock_update = False

                # GATING LOGIC (Revised 2025-12-12):
                # We want to allow updates if:
                # 1. We have robust multi-station fusion (n>=3, uncertainty<=3ms)
                # 2. We have a decent dual-station consensus (n=2, uncertainty<=1ms)
                # 3. We have ONE SINGLE high-quality station (uncertainty<=1.0ms, SNR high) 
                #    - This enables users with only WWV or CHU to still converge!
                
                if fusion_result.n_broadcasts >= 3:
                     # Robust fusion
                     if fusion_result.uncertainty_ms <= 3.0:
                         allow_clock_update = True
                elif fusion_result.n_broadcasts == 2:
                     # Dual station
                     if fusion_result.uncertainty_ms <= 1.0:
                         allow_clock_update = True
                elif fusion_result.n_broadcasts == 1:
                     # Single station - must be very high quality
                     # e.g. WWV local graping or strong CHU
                     # Require uncertainty < 1.0ms (implies very stable, clean signal)
                     # BUG FIX (2025-12-12): Relax grade requirement since single station
                     # often gets 'C' or 'D' purely due to count < 3.
                     # Uncertainty is the real metric here.
                     if fusion_result.uncertainty_ms <= 1.0:
                         allow_clock_update = True
                         logger.info(f"  Fusion: Allowing single-station update (U={fusion_result.uncertainty_ms:.2f}ms)")

                if allow_clock_update:
                    raw_d_clock = fusion_result.d_clock_ms
                    convergence = self.clock_convergence.process_measurement(
                        d_clock_ms=raw_d_clock,
                        timestamp=time.time(),
                        measurement_noise_ms=fusion_result.uncertainty_ms
                    )

                    self.d_clock_ms = convergence.d_clock_ms
                    self.d_clock_uncertainty_ms = convergence.uncertainty_ms
                    self.last_fusion = fusion_result
                    applied_clock_update = True

                    logger.info(
                        f"  Kalman: raw={raw_d_clock:+.2f}ms → filtered={convergence.d_clock_ms:+.2f}ms "
                        f"±{convergence.uncertainty_ms:.2f}ms [{convergence.state.value}] "
                        f"innov={convergence.innovation_ms:+.2f}ms"
                    )
                else:
                    logger.warning(
                        f"  Fusion gated: used={fusion_result.n_broadcasts} "
                        f"uncertainty={fusion_result.uncertainty_ms:.2f}ms grade={fusion_result.quality_grade}"
                    )
                
                # Record to fusion history for web GUI
                self.fusion_history.append({
                    'timestamp': time.time(),
                    'd_clock_fused_ms': fusion_result.d_clock_ms,
                    'd_clock_raw_ms': fusion_result.d_clock_raw_ms,
                    'uncertainty_ms': fusion_result.uncertainty_ms,
                    'n_broadcasts': fusion_result.n_broadcasts,
                    'quality_grade': fusion_result.quality_grade
                })
                
                # Call callback if registered (for web server SSE)
                if self.on_fusion_update:
                    try:
                        self.on_fusion_update(fusion_result)
                    except Exception as e:
                        logger.debug(f"Fusion callback error: {e}")
                
                logger.info(f"  Slow Loop fused: D_clock={fusion_result.d_clock_ms:+.3f}ms "
                           f"±{fusion_result.uncertainty_ms:.3f}ms from {fusion_result.n_broadcasts} broadcasts "
                           f"[grade {fusion_result.quality_grade}]")
        
        # Create and publish timing solution
        if results:
            solution = TimingSolution(
                timestamp=time.time(),
                rtp_anchor=0,  # TODO
                utc_ref=minute * 60,
                d_clock_ms=self.d_clock_ms,
                d_clock_uncertainty_ms=self.d_clock_uncertainty_ms,
                clock_drift_ppm=self.clock_drift_ppm,
                status=self.state.value,
                primary_station=list(results.values())[0]['station'] if results else 'UNKNOWN',
                propagation_mode=list(results.values())[0]['propagation_mode'] if results else 'UNKNOWN',
                channels_contributing=len(results),
                discrimination={
                    name: {'snr': r['snr_db'], 'confidence': r['confidence']}
                    for name, r in results.items()
                }
            )
            
            try:
                self._publish_shm(solution)
            except Exception as e:
                logger.debug(f"SHM publish failed: {e}")
            
            # Update Chrony if enabled
            if self.enable_chrony and self.chrony_shm and applied_clock_update:
                self._update_chrony()
            
            self._save_state()
    
    def _extract_all_broadcasts(self, freq_name: str, phase2_result, solution, time_snap) -> Dict[str, Dict]:
        """
        Extract ALL detected broadcasts from a single frequency.
        
        On shared frequencies (2.5, 5, 10, 15 MHz), both WWV and WWVH may be detected.
        This method extracts timing for ALL detected stations, not just the dominant one.
        
        Args:
            freq_name: Frequency channel name (e.g., "WWV 10 MHz")
            phase2_result: Complete Phase2Result
            solution: TransmissionTimeSolution (for dominant station)
            time_snap: TimeSnapResult with per-station timing
            
        Returns:
            Dict mapping broadcast keys to result dicts
            e.g., {"WWV 10 MHz": {...}, "WWVH 10 MHz": {...}}
        """
        broadcasts = {}
        
        # Extract frequency from channel name
        freq_mhz = solution.frequency_mhz if solution else 10.0
        
        # Shared frequencies where both WWV and WWVH broadcast
        SHARED_FREQS = {2.5, 5.0, 10.0, 15.0}
        is_shared = freq_mhz in SHARED_FREQS
        
        # CHU frequencies
        CHU_FREQS = {3.33, 7.85, 14.67}
        is_chu = freq_mhz in CHU_FREQS
        
        # Get propagation mode from dominant solution
        prop_mode = solution.propagation_mode if solution else '2F'
        
        # Helper to get proper per-station propagation delay
        def get_station_delay(station: str) -> float:
            """Get propagation delay for specific station using proper geometry."""
            if self._propagation_solver:
                return self._propagation_solver.get_station_propagation_delay(
                    station=station,
                    frequency_mhz=freq_mhz,
                    mode=prop_mode
                )
            else:
                # Fallback to dominant solution delay (less accurate)
                return solution.t_propagation_ms if solution else 0.0
        
        # Extract WWV timing if detected
        if time_snap and time_snap.wwv_detected and time_snap.wwv_timing_ms is not None:
            wwv_key = f"WWV {freq_mhz} MHz"
            wwv_prop_delay = get_station_delay('WWV')
            # D_clock = arrival_time - propagation_delay (back-calculate to UTC(NIST))
            wwv_d_clock = time_snap.wwv_timing_ms - wwv_prop_delay
            broadcasts[wwv_key] = {
                'station': 'WWV',
                'propagation_mode': prop_mode,
                'propagation_delay_ms': wwv_prop_delay,
                'snr_db': time_snap.wwv_snr_db or 0.0,
                'confidence': phase2_result.confidence if time_snap.wwv_snr_db and time_snap.wwv_snr_db > 3 else 0.1,
                'd_clock_ms': wwv_d_clock,
                'timing_ms': time_snap.wwv_timing_ms
            }
        
        # Extract WWVH timing if detected (only on shared frequencies)
        if is_shared and time_snap and time_snap.wwvh_detected and time_snap.wwvh_timing_ms is not None:
            wwvh_key = f"WWVH {freq_mhz} MHz"
            wwvh_prop_delay = get_station_delay('WWVH')
            # D_clock = arrival_time - propagation_delay (back-calculate to UTC(NIST))
            wwvh_d_clock = time_snap.wwvh_timing_ms - wwvh_prop_delay
            broadcasts[wwvh_key] = {
                'station': 'WWVH',
                'propagation_mode': prop_mode,
                'propagation_delay_ms': wwvh_prop_delay,
                'snr_db': time_snap.wwvh_snr_db or 0.0,
                'confidence': phase2_result.confidence * 0.8 if time_snap.wwvh_snr_db and time_snap.wwvh_snr_db > 3 else 0.05,
                'd_clock_ms': wwvh_d_clock,
                'timing_ms': time_snap.wwvh_timing_ms
            }
        
        # Extract CHU timing if detected
        if is_chu and time_snap and time_snap.chu_detected and time_snap.chu_timing_ms is not None:
            chu_key = f"CHU {freq_mhz} MHz"
            chu_prop_delay = get_station_delay('CHU')
            # D_clock = arrival_time - propagation_delay (back-calculate to UTC(NRC))
            chu_d_clock = time_snap.chu_timing_ms - chu_prop_delay
            broadcasts[chu_key] = {
                'station': 'CHU',
                'propagation_mode': prop_mode,
                'propagation_delay_ms': chu_prop_delay,
                'snr_db': time_snap.chu_snr_db or 0.0,
                'confidence': phase2_result.confidence if time_snap.chu_snr_db and time_snap.chu_snr_db > 3 else 0.1,
                'd_clock_ms': chu_d_clock,
                'timing_ms': time_snap.chu_timing_ms
            }
        
        # Fallback: if no broadcasts extracted, use the dominant solution
        if not broadcasts:
            fallback_station = solution.station if solution else 'WWV'
            fallback_delay = get_station_delay(fallback_station)
            broadcasts[freq_name] = {
                'station': fallback_station,
                'propagation_mode': prop_mode,
                'propagation_delay_ms': fallback_delay,
                'snr_db': 0.0,
                'confidence': phase2_result.confidence,
                'd_clock_ms': phase2_result.d_clock_ms
            }
        
        return broadcasts
    
    def _fuse_broadcasts(self, results: Dict[str, Dict]) -> Optional[FusionResult]:
        """
        Fuse D_clock from all broadcasts using improved algorithm.
        
        Implements:
        1. Per-broadcast calibration with EMA learning
        2. Weight calculation (grade × mode × SNR)
        3. MAD-based outlier rejection
        4. Quality grading
        
        Args:
            results: Dict of broadcast results with d_clock_ms, station, etc.
            
        Returns:
            FusionResult with fused D_clock and quality metrics
        """
        if not results:
            return None
        
        # Weight factors from previous grape-recorder
        GRADE_WEIGHTS = {'A': 1.0, 'B': 0.8, 'C': 0.5, 'D': 0.2}
        MODE_WEIGHTS = {
            '1E': 1.0, '1F': 0.9, '2F': 0.7, '3F': 0.5, 'GW': 1.0,
            '2E': 0.85, '3E': 0.6, 'UNKNOWN': 0.3
        }
        
        n_total_candidates = len(results)
        n_prefilter_rejected = 0

        # Collect measurements with weights
        measurements = []  # [(broadcast_key, d_clock_raw, d_clock_calibrated, weight, station)]
        
        for broadcast_key, r in results.items():
            if r['d_clock_ms'] is None:
                continue
            
            station = r.get('station', 'UNKNOWN')
            freq_mhz = r.get('frequency_mhz', 0.0)
            if freq_mhz == 0.0:
                # Parse from broadcast_key (e.g., "WWV 10 MHz" or "WWV 10.0 MHz")
                try:
                    parts = broadcast_key.split()
                    freq_mhz = float(parts[1])
                except:
                    freq_mhz = 10.0
            
            d_clock_raw = r['d_clock_ms']
            
            # FIX 2 (2025-12-11): Pre-filter physically implausible measurements
            # D_clock should be within ±100ms for a GPSDO-disciplined system.
            # However, during acquisition (unlocked), we must allow large offsets.
            if self.clock_convergence.is_locked and abs(d_clock_raw) > 100.0:
                logger.debug(f"  {broadcast_key}: REJECTED (d_clock={d_clock_raw:+.1f}ms outside ±100ms)")
                n_prefilter_rejected += 1
                
                # Monitor systematic rejection (wrong basin)
                self.consecutive_rejections += 1
                if self.consecutive_rejections >= 5:
                    logger.warning(f"  Fusion: {self.consecutive_rejections} consecutive rejections - resetting Kalman to re-acquire")
                    self.clock_convergence.reset(initial_offset_ms=d_clock_raw)
                    self.consecutive_rejections = 0
                continue
            
            # Reset rejection counter if we found a valid candidate
            self.consecutive_rejections = 0

            # Apply per-broadcast calibration (station+frequency).
            # This absorbs systematic biases that remain after propagation modeling
            # (e.g., CHU cross-frequency offsets, matched-filter group delay, residual 1/f^2 errors).
            cal_key = self._get_broadcast_key(station, freq_mhz)
            cal = self.calibration.get(cal_key)
            cal_offset_ms = 0.0
            if cal and cal.n_samples >= 10:
                if abs(float(cal.offset_ms)) <= 500.0:
                    cal_offset_ms = float(cal.offset_ms)
                else:
                    logger.warning(
                        f"  {broadcast_key}: ignoring invalid calibration offset {float(cal.offset_ms):+.3f}ms"
                    )
            d_clock_calibrated = d_clock_raw + cal_offset_ms
            
            # Calculate weight
            confidence = r.get('confidence', 0.5)
            mode = r.get('propagation_mode', 'UNKNOWN')
            snr_db = r.get('snr_db', 0.0)
            grade = r.get('quality_grade', 'C')
            
            # Weight = confidence × grade × mode × SNR_factor
            grade_w = GRADE_WEIGHTS.get(grade, 0.2)
            mode_w = MODE_WEIGHTS.get(mode, 0.5)
            
            if snr_db > 10:
                snr_w = 1.0
            elif snr_db > 5:
                snr_w = 0.8
            elif snr_db > 0:
                snr_w = 0.5
            else:
                snr_w = 0.3

            # Penalize measurements whose calibration is still uncertain.
            # If we don't have enough history, cal.uncertainty_ms stays large.
            cal_uncertainty_w = 1.0
            if cal and cal.n_samples >= 10:
                # Map uncertainty to (0.2..1.0) weight factor.
                # <=1ms => 1.0, 2ms => 0.8, 5ms => 0.4, >=10ms => 0.2
                cu = max(0.0, float(cal.uncertainty_ms))
                cal_uncertainty_w = max(0.2, min(1.0, 1.0 - (cu / 12.5)))
            else:
                cal_uncertainty_w = 0.3

            weight = max(0.01, confidence * grade_w * mode_w * snr_w * cal_uncertainty_w)
            
            measurements.append((broadcast_key, d_clock_raw, d_clock_calibrated, weight, station, freq_mhz))
            
            # Update calibration history (for future use when propagation model is tuned)
            cal_key = self._get_broadcast_key(station, freq_mhz)
            if cal_key not in self.calibration_history:
                self.calibration_history[cal_key] = deque(maxlen=self.calibration_history_max)
            self.calibration_history[cal_key].append(d_clock_raw)
        
        if len(measurements) < 2:
            return None
            
        # FIX (2025-12-12): Global Anchor Enforcement
        # Since all channels are sampled simultaneously by the same ADC, 
        # relative timing differences > 100ms are physically impossible.
        # We find the single strongest signal (SNR) and enforce its reality on all others.
        
        # Sort by SNR (assuming SNR is stored/available, or calculate from weight)
        # measurements tuple: (key, d_raw, d_cal, weight, station, freq)
        # We don't have raw SNR in the tuple! I need to trace back to 'results'.
        # Actually, 'weight' includes SNR factor. Max weight is a good proxy for "best signal".
        
        best_measurement = max(measurements, key=lambda m: m[3]) # Sort by weight
        best_d_clock = best_measurement[2]
        best_weight = best_measurement[3]
        
        # Only enforce if the anchor is reasonably strong (weight > 0.01 implies decent SNR/Conf)
        # lowered from 0.1 to 0.01 (2025-12-12) to catch Grade D measurements
        if best_weight > 0.01:
            filtered_measurements = []
            n_rejected_anchor = 0
            
            for m in measurements:
                # Allowed deviation: 100ms (generous propagation variance + processing jitter)
                if abs(m[2] - best_d_clock) <= 100.0:
                    filtered_measurements.append(m)
                else:
                    n_rejected_anchor += 1
            
            if n_rejected_anchor > 0:
                logger.info(f"  Global Anchor Rejection: Rejected {n_rejected_anchor} broadcasters "
                           f"deviating >100ms from anchor {best_measurement[0]} (d={best_d_clock:+.1f}ms)")
            
            measurements = filtered_measurements
            
        if len(measurements) < 2:
             # If we reduced to 1 measurement, pass it through (gated by uncertainty later)
             pass

        # MAD-based outlier rejection

        d_calibrated = np.array([m[2] for m in measurements])
        weights = np.array([m[3] for m in measurements])
        
        n_rejected = 0
        if len(measurements) >= 4:
            # Weighted median
            sorted_idx = np.argsort(d_calibrated)
            sorted_d = d_calibrated[sorted_idx]
            sorted_w = weights[sorted_idx]
            cumsum = np.cumsum(sorted_w)
            median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
            weighted_median = sorted_d[min(median_idx, len(sorted_d)-1)]
            
            # MAD (Median Absolute Deviation) × 1.4826 to scale to std
            deviations = np.abs(d_calibrated - weighted_median)
            mad = np.median(deviations) * 1.4826
            
            if mad < 0.1:
                mad = 0.1  # Minimum to avoid division by zero
            
            # Reject outliers > 2 MAD OR > 10ms from median (hard cap)
            # The hard cap catches cases where MAD is inflated by multiple outliers
            mad_threshold = min(2.0 * mad, 10.0)  # At most 10ms deviation allowed
            keep_mask = deviations < mad_threshold
            
            if np.sum(keep_mask) >= 2:
                n_rejected = len(measurements) - np.sum(keep_mask)
                if n_rejected > 0:
                    logger.info(f"  MAD outlier rejection: {n_rejected} broadcasts "
                               f"(median={weighted_median:.2f}ms, MAD={mad:.2f}ms)")
                measurements = [m for m, keep in zip(measurements, keep_mask) if keep]
                d_calibrated = np.array([m[2] for m in measurements])
                weights = np.array([m[3] for m in measurements])
        
        # Weighted mean
        total_weight = np.sum(weights)
        if total_weight <= 0:
            return None
        
        fused_d_clock = np.sum(weights * d_calibrated) / total_weight
        
        # Raw (uncalibrated) mean
        d_raw = np.array([m[1] for m in measurements])
        raw_mean = np.mean(d_raw)
        
        # Weighted standard deviation
        weighted_var = np.sum(weights * (d_calibrated - fused_d_clock)**2) / total_weight
        uncertainty = np.sqrt(weighted_var) if weighted_var > 0 else 1.0
        
        # Station counts
        stations = [m[4] for m in measurements]
        wwv_count = sum(1 for s in stations if s == 'WWV')
        wwvh_count = sum(1 for s in stations if s == 'WWVH')
        chu_count = sum(1 for s in stations if s == 'CHU')
        
        # Quality grade (from grape-recorder)
        n_broadcasts = len(measurements)
        if n_broadcasts >= 8 and uncertainty < 0.5:
            grade = 'A'
        elif n_broadcasts >= 5 and uncertainty < 1.0:
            grade = 'B'
        elif n_broadcasts >= 3 and uncertainty < 2.0:
            grade = 'C'
        else:
            grade = 'D'
        
        # Update calibration (EMA learning)
        self._update_calibration(measurements)

        if grade == 'D' or uncertainty > 3.0:
            # Log the actual measurements that drove the fused value.
            # This is critical for diagnosing instability: are we mixing stations,
            # using low-SNR detections, or seeing propagation model mismatch?
            try:
                meas_lines = []
                for broadcast_key, d_clock_raw, d_clock_calibrated, weight, station, freq_mhz in measurements:
                    r = results.get(broadcast_key, {})
                    timing_ms = r.get('timing_ms', None)
                    prop_ms = r.get('propagation_delay_ms', None)

                    cal_key = self._get_broadcast_key(station, freq_mhz)
                    cal = self.calibration.get(cal_key)
                    cal_offset_ms = float(cal.offset_ms) if (cal and cal.n_samples >= 10) else 0.0
                    meas_lines.append(
                        f"{broadcast_key} {station} {freq_mhz:.2f}MHz: "
                        f"timing={timing_ms if timing_ms is not None else 'None'}ms "
                        f"prop={prop_ms if prop_ms is not None else 'None'}ms "
                        f"raw={d_clock_raw:+.3f}ms cal={cal_offset_ms:+.3f}ms "
                        f"d_clock={d_clock_calibrated:+.3f}ms w={weight:.3f}"
                    )
                logger.info("  Fusion inputs (post-filter): " + "; ".join(meas_lines))
            except Exception as e:
                logger.debug(f"Fusion measurement logging failed: {e}")
        
        return FusionResult(
            d_clock_ms=fused_d_clock,
            d_clock_raw_ms=raw_mean,
            uncertainty_ms=uncertainty,
            n_broadcasts=n_broadcasts,
            n_outliers_rejected=n_rejected,
            quality_grade=grade,
            n_total_candidates=n_total_candidates,
            n_prefilter_rejected=n_prefilter_rejected,
            wwv_count=wwv_count,
            wwvh_count=wwvh_count,
            chu_count=chu_count
        )
    
    def _update_calibration(self, measurements: List):
        """
        Update per-broadcast calibration using EMA.
        
        Calibration offset brings each broadcast's mean D_clock toward the
        current global clock estimate (self.d_clock_ms).

        This is important because D_clock is not guaranteed to be 0ms.
        What we actually need is cross-broadcast CONSISTENCY so we can fuse
        multiple frequencies/stations without inflating uncertainty.
        Uses exponential moving average with α = max(0.5, 20/n_samples).
        """
        # Only learn calibration when we have a reasonably stable reference.
        # If we learn too early, we will bake transient acquisition errors into
        # per-broadcast offsets.
        reference_d_clock_ms = float(self.d_clock_ms)
        if not self.clock_convergence.is_locked and self.d_clock_uncertainty_ms > 5.0:
            return

        if abs(reference_d_clock_ms) > 200.0:
            return

        for broadcast_key, d_clock_raw, _, _, station, freq_mhz in measurements:
            cal_key = self._get_broadcast_key(station, freq_mhz)
            history = self.calibration_history.get(cal_key)
            
            if not history or len(history) < 10:
                continue
            
            # Recent measurements
            recent = list(history)[-30:]
            mean_d_clock = np.mean(recent)
            std_d_clock = np.std(recent) if len(recent) > 1 else 1.0
            
            # New offset should bring mean to current global estimate
            new_offset = reference_d_clock_ms - mean_d_clock

            if abs(float(new_offset)) > 500.0:
                continue
            
            # EMA update
            old_cal = self.calibration.get(cal_key)
            if old_cal and old_cal.n_samples > 0:
                alpha = max(0.5, 20.0 / old_cal.n_samples)
                new_offset = alpha * new_offset + (1 - alpha) * old_cal.offset_ms

            if abs(float(new_offset)) > 500.0:
                continue
            
            self.calibration[cal_key] = BroadcastCalibration(
                station=station,
                frequency_mhz=freq_mhz,
                offset_ms=new_offset,
                uncertainty_ms=std_d_clock,
                n_samples=len(history),
                last_updated=time.time()
            )
        
        # Save periodically (every 10 updates)
        if sum(c.n_samples for c in self.calibration.values()) % 10 == 0:
            self._save_calibration()
    
    def _update_chrony(self):
        """Update Chrony SHM with current D_clock.
        
        D_clock = system_time - reference_time (positive if system is fast)
        We pass: reference_time = system_time - D_clock
        
        The chrony_shm.update() method expects:
        - reference_time: The "true" time according to WWV/CHU
        - system_time: When the measurement was taken
        """
        if not self.chrony_shm:
            logger.warning("Chrony SHM not connected")
            return
        
        try:
            now = time.time()
            # D_clock is how much system clock is ahead of UTC (in ms)
            # reference_time is what WWV says the time is
            offset_sec = self.d_clock_ms / 1000.0
            reference_time = now - offset_sec
            
            self.chrony_shm.update(reference_time, now, precision=-13)  # ~100µs with GPSDO
            self.stats['chrony_updates'] += 1
            logger.info(f"  Chrony updated: D_clock={self.d_clock_ms:+.2f}ms (update #{self.stats['chrony_updates']})")
        except Exception as e:
            logger.error(f"Chrony update failed: {e}")
    
    def start(self):
        """Start the live time engine."""
        if self.running:
            logger.warning("Engine already running")
            return
        
        logger.info("Starting LiveTimeEngine...")
        self.stats['start_time'] = time.time()
        self.running = True
        
        # Initialize channel buffers
        self._init_channel_buffers()
        
        # Load persisted state
        has_state = self._load_state()
        self.state = EngineState.TRACKING if has_state else EngineState.ACQUIRING
        
        # Initialize Chrony if enabled
        if self.enable_chrony:
            try:
                from ..output.chrony_shm import ChronySHM
                self.chrony_shm = ChronySHM(unit=self.chrony_unit)
                self.chrony_shm.connect()
                logger.info("Chrony SHM connected")
            except Exception as e:
                logger.error(f"Failed to initialize Chrony: {e}")
        
        # Initialize RTP streams (RadiodStream instances are started inside)
        self._init_rtp_receiver()
        
        # Start processing threads
        self.fast_loop_thread = threading.Thread(
            target=self._fast_loop, 
            name="FastLoop",
            daemon=True
        )
        self.slow_loop_thread = threading.Thread(
            target=self._slow_loop,
            name="SlowLoop", 
            daemon=True
        )
        
        self.fast_loop_thread.start()
        self.slow_loop_thread.start()
        
        logger.info("LiveTimeEngine started")
        logger.info(f"  State: {self.state.value}")
        logger.info(f"  Fast Loop: waiting for T=:01")
        logger.info(f"  Slow Loop: waiting for T=:06")
    
    def stop(self):
        """Stop the live time engine."""
        logger.info("Stopping LiveTimeEngine...")
        self.running = False
        
        # Stop RadiodStream instances
        if hasattr(self, '_streams'):
            for stream in self._streams:
                try:
                    stream.stop()
                except Exception:
                    pass
        
        # Legacy: stop old-style RTP receiver if present
        if hasattr(self, 'rtp_receiver') and self.rtp_receiver:
            self.rtp_receiver.stop()
        
        # Wait for threads
        if self.fast_loop_thread:
            self.fast_loop_thread.join(timeout=2)
        if self.slow_loop_thread:
            self.slow_loop_thread.join(timeout=2)
        
        # Disconnect Chrony
        if self.chrony_shm:
            self.chrony_shm.disconnect()
        
        # Save state
        self._save_state()
        
        # Log stats
        uptime = time.time() - self.stats['start_time']
        logger.info("LiveTimeEngine stopped")
        logger.info(f"  Uptime: {uptime:.1f}s")
        logger.info(f"  Fast loops: {self.stats['fast_loop_count']}")
        logger.info(f"  Slow loops: {self.stats['slow_loop_count']}")
        logger.info(f"  Chrony updates: {self.stats['chrony_updates']}")
    
    def run(self):
        """Run the engine (blocking)."""
        import signal
        
        # Handle shutdown signals
        def handle_signal(signum, frame):
            logger.info(f"Received signal {signum}")
            self.stop()
        
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)
        
        self.start()
        
        # Block until stopped
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
