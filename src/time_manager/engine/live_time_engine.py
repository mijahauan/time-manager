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
    
    # Per-station breakdown
    wwv_count: int = 0
    wwvh_count: int = 0
    chu_count: int = 0


@dataclass 
class ChannelBuffer:
    """
    Per-channel ring buffer and full-minute buffer.
    
    Ring buffer: Keeps ±1.5s around minute boundary for Fast Loop
    Full buffer: Accumulates entire minute for Slow Loop
    """
    channel_name: str
    ssrc: int
    sample_rate: int = 20000
    
    # Ring buffer for Fast Loop (3 seconds = 60,000 samples)
    ring_size: int = 60000
    ring_buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.complex64))
    ring_write_pos: int = 0
    ring_start_rtp: int = 0
    
    # Full minute buffer for Slow Loop (60 seconds = 1,200,000 samples)
    full_size: int = 1200000
    full_buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.complex64))
    full_write_pos: int = 0
    full_start_rtp: int = 0
    full_ready: bool = False  # Set when minute is complete
    
    # RTP tracking
    last_rtp_timestamp: int = 0
    last_wallclock: float = 0.0
    packets_received: int = 0
    
    def __post_init__(self):
        """Initialize buffers."""
        self.ring_buffer = np.zeros(self.ring_size, dtype=np.complex64)
        self.full_buffer = np.zeros(self.full_size, dtype=np.complex64)
    
    def add_samples(self, samples: np.ndarray, rtp_timestamp: int, wallclock: float):
        """
        Add samples to buffers.
        
        Ring buffer: Only keep samples near minute boundary (±1.5s)
        Full buffer: Accumulate all samples for the minute
        """
        self.last_rtp_timestamp = rtp_timestamp
        self.last_wallclock = wallclock
        self.packets_received += 1
        
        n_samples = len(samples)
        
        # Add to full buffer (always, until full)
        if self.full_write_pos + n_samples <= self.full_size:
            self.full_buffer[self.full_write_pos:self.full_write_pos + n_samples] = samples
            self.full_write_pos += n_samples
            
            if self.full_write_pos == 0:
                self.full_start_rtp = rtp_timestamp
        
        # Ring buffer: circular write
        # Calculate position in ring
        write_start = self.ring_write_pos % self.ring_size
        
        if write_start + n_samples <= self.ring_size:
            # Simple case: fits without wrap
            self.ring_buffer[write_start:write_start + n_samples] = samples
        else:
            # Wrap around
            first_part = self.ring_size - write_start
            self.ring_buffer[write_start:] = samples[:first_part]
            self.ring_buffer[:n_samples - first_part] = samples[first_part:]
        
        self.ring_write_pos += n_samples
        
        # Track ring buffer start
        if self.ring_write_pos <= self.ring_size:
            if self.ring_start_rtp == 0:
                self.ring_start_rtp = rtp_timestamp
    
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
    
    def get_full_samples(self) -> tuple[np.ndarray, int]:
        """
        Get full minute samples (for Slow Loop).
        
        Returns:
            (samples, start_rtp_timestamp)
        """
        return self.full_buffer[:self.full_write_pos].copy(), self.full_start_rtp
    
    def reset_for_new_minute(self):
        """Reset buffers for new minute."""
        # Keep ring buffer rolling
        self.ring_write_pos = 0
        self.ring_start_rtp = 0
        
        # Clear full buffer
        self.full_write_pos = 0
        self.full_start_rtp = 0
        self.full_ready = False


class LiveTimeEngine:
    """
    Live timing engine with Twin-Stream architecture.
    
    Subscribes to RTP multicast and performs real-time timing extraction
    with Fast Loop (tone detection) and Slow Loop (discrimination).
    """
    
    STATE_FILE = "/var/lib/grape-recorder/state/time_state.json"
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
        self.calibration_file = Path("/var/lib/grape-recorder/state/broadcast_calibration.json")
        self._load_calibration()
        
        # History for calibration learning (broadcast_key -> list of d_clock values)
        self.calibration_history: Dict[str, deque] = {}
        self.calibration_history_max = 100  # Keep last N measurements per broadcast
        
        # Last fusion result
        self.last_fusion: Optional[FusionResult] = None
        
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
                
                for broadcast_key, cal_data in data.items():
                    # Parse station and frequency from key
                    parts = broadcast_key.rsplit('_', 1)
                    station = parts[0] if len(parts) > 1 else broadcast_key
                    freq = float(parts[1]) if len(parts) > 1 else 0.0
                    
                    self.calibration[broadcast_key] = BroadcastCalibration(
                        station=station,
                        frequency_mhz=cal_data.get('frequency_mhz', freq),
                        offset_ms=cal_data.get('offset_ms', 0.0),
                        uncertainty_ms=cal_data.get('uncertainty_ms', 10.0),
                        n_samples=cal_data.get('n_samples', 0),
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
    
    def _stream_callback(self, channel_name: str):
        """
        Create callback for RadiodStream.
        
        RadiodStream delivers (samples: np.ndarray, quality: StreamQuality).
        We adapt this to add samples to our channel buffer.
        """
        def callback(samples: np.ndarray, quality):
            if channel_name not in self.channel_buffers:
                return
            
            # Get timing info from quality metrics
            # RadiodStream provides continuous sample stream with quality tracking
            current_time = time.time()
            
            # Use quality.total_samples as pseudo-RTP timestamp
            # (RadiodStream handles resequencing internally)
            rtp_timestamp = quality.total_samples if hasattr(quality, 'total_samples') else 0
            
            # Add to buffer
            self.channel_buffers[channel_name].add_samples(
                samples, rtp_timestamp, current_time
            )
        
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
            discover_channels, RadiodStream, RadiodControl, allocate_ssrc
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
        
        # Build frequency -> ChannelInfo lookup
        freq_to_channel = {}
        for ssrc, ch_info in discovered.items():
            freq_hz = int(round(ch_info.frequency))
            freq_to_channel[freq_hz] = ch_info
        
        logger.info(f"  Found {len(discovered)} existing channels")
        
        # Get RadiodControl for creating/configuring channels
        control = RadiodControl(self.status_address)
        
        # Ensure each time signal frequency has a channel
        self.channels = []
        self._streams = []  # RadiodStream instances
        
        for ch_spec in TIME_SIGNAL_FREQS:
            freq_hz = int(ch_spec['frequency_hz'])
            name = ch_spec['description']
            
            if freq_hz in freq_to_channel:
                ch_info = freq_to_channel[freq_hz]
                
                # Check if destination matches
                if ch_info.multicast_address != our_destination:
                    # Reconfigure to our destination
                    logger.info(f"  ⚙ {name}: reconfiguring to {our_destination}")
                    try:
                        control.create_channel(
                            frequency_hz=freq_hz,
                            preset=CHANNEL_DEFAULTS['preset'],
                            sample_rate=CHANNEL_DEFAULTS['sample_rate'],
                            destination=our_destination,
                            ssrc=ch_info.ssrc
                        )
                        # Re-discover to get updated info
                        import time as time_mod
                        time_mod.sleep(0.3)
                        discovered = discover_channels(self.status_address, listen_duration=1.0)
                        ch_info = discovered.get(ch_info.ssrc, ch_info)
                    except Exception as e:
                        logger.warning(f"  Failed to reconfigure {name}: {e}")
                
                logger.info(f"  ✓ {name}: SSRC={ch_info.ssrc}")
            else:
                # Create new channel
                logger.info(f"  ➕ {name}: creating...")
                try:
                    ssrc = allocate_ssrc()
                    control.create_channel(
                        frequency_hz=freq_hz,
                        preset=CHANNEL_DEFAULTS['preset'],
                        sample_rate=CHANNEL_DEFAULTS['sample_rate'],
                        destination=our_destination,
                        ssrc=ssrc
                    )
                    import time as time_mod
                    time_mod.sleep(0.3)
                    discovered = discover_channels(self.status_address, listen_duration=1.0)
                    ch_info = discovered.get(ssrc)
                    if ch_info:
                        logger.info(f"    Created SSRC={ssrc}")
                    else:
                        logger.warning(f"    Channel creation unverified")
                        continue
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
            callback = self._stream_callback(name)
            
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
                
                # FIX Issue 1: Use smoothed RTP-to-System offset (eliminates host jitter)
                # This is the key to microsecond precision with GPSDO
                if name in self.rtp_system_offset:
                    # Use the "steel ruler": RTP time + smoothed offset
                    buffer_start_time = (start_rtp / self.sample_rate) + self.rtp_system_offset[name]
                else:
                    # Fallback to raw wallclock (jittery but safe)
                    buffer_start_time = buffer.last_wallclock - (len(samples) / self.sample_rate)
                
                # Run tone detection with narrow window (Fast Loop uses previous state)
                detections = detector.process_samples(
                    timestamp=buffer_start_time,
                    samples=samples,
                    rtp_timestamp=start_rtp,
                    original_sample_rate=self.sample_rate,
                    search_window_ms=100.0,  # Narrow ±100ms window
                    expected_offset_ms=state.propagation_delay_ms  # Expected arrival
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
        Slow Loop thread - runs at T=:02 each minute.
        
        Uses full minute buffer for complete discrimination and
        state update for next Fast Loop iteration.
        """
        logger.info("Slow Loop thread started")
        
        while self.running:
            try:
                # Wait for second :02 of the minute
                now = time.time()
                current_second = now % 60
                
                # Calculate time to :02
                if current_second < 2:
                    wait_time = 2 - current_second
                else:
                    wait_time = 62 - current_second
                
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
            # Get full minute samples
            samples, start_rtp = buffer.get_full_samples()
            
            if len(samples) < self.sample_rate * 30:  # Need at least 30 seconds
                logger.debug(f"  {name}: Insufficient samples ({len(samples)})")
                buffer.reset_for_new_minute()
                continue
            
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
                
                # Calculate system time for this minute
                system_time = float(minute * 60)
                
                # Run full Phase 2 analysis
                phase2_result = engine.process_minute(
                    iq_samples=samples,
                    system_time=system_time,
                    rtp_timestamp=start_rtp
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
            
            # Reset buffer for next minute
            buffer.reset_for_new_minute()
        
        # Transition from ACQUIRING to TRACKING
        if self.state == EngineState.ACQUIRING and results:
            self.state = EngineState.TRACKING
            logger.info("State transition: ACQUIRING -> TRACKING")
        
        # Fuse D_clock from ALL broadcasts (up to 13) using improved algorithm
        if results:
            fusion_result = self._fuse_broadcasts(results)
            
            if fusion_result:
                self.d_clock_ms = fusion_result.d_clock_ms
                self.d_clock_uncertainty_ms = fusion_result.uncertainty_ms
                self.last_fusion = fusion_result
                
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
            
            self._publish_shm(solution)
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
        
        # Base propagation delay from the dominant solution
        base_prop_delay = solution.t_propagation_ms if solution else 0.0
        prop_mode = solution.propagation_mode if solution else '2F'
        
        # Extract WWV timing if detected
        if time_snap and time_snap.wwv_detected and time_snap.wwv_timing_ms is not None:
            wwv_key = f"WWV {freq_mhz} MHz"
            # D_clock = arrival_time - propagation_delay
            # timing_ms is offset from minute boundary
            wwv_d_clock = time_snap.wwv_timing_ms - base_prop_delay
            broadcasts[wwv_key] = {
                'station': 'WWV',
                'propagation_mode': prop_mode,
                'propagation_delay_ms': base_prop_delay,
                'snr_db': time_snap.wwv_snr_db or 0.0,
                'confidence': phase2_result.confidence if time_snap.wwv_snr_db and time_snap.wwv_snr_db > 3 else 0.1,
                'd_clock_ms': wwv_d_clock,
                'timing_ms': time_snap.wwv_timing_ms
            }
        
        # Extract WWVH timing if detected (only on shared frequencies)
        if is_shared and time_snap and time_snap.wwvh_detected and time_snap.wwvh_timing_ms is not None:
            wwvh_key = f"WWVH {freq_mhz} MHz"
            # WWVH has different propagation path - use scaled delay
            # WWVH is ~6600km from receiver vs ~1100km for WWV (roughly 6x)
            wwvh_prop_delay = base_prop_delay * 2.5  # Approximate scaling
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
            chu_d_clock = time_snap.chu_timing_ms - base_prop_delay
            broadcasts[chu_key] = {
                'station': 'CHU',
                'propagation_mode': prop_mode,
                'propagation_delay_ms': base_prop_delay,
                'snr_db': time_snap.chu_snr_db or 0.0,
                'confidence': phase2_result.confidence if time_snap.chu_snr_db and time_snap.chu_snr_db > 3 else 0.1,
                'd_clock_ms': chu_d_clock,
                'timing_ms': time_snap.chu_timing_ms
            }
        
        # Fallback: if no broadcasts extracted, use the dominant solution
        if not broadcasts:
            broadcasts[freq_name] = {
                'station': solution.station if solution else 'UNKNOWN',
                'propagation_mode': prop_mode,
                'propagation_delay_ms': base_prop_delay,
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
            
            # Apply calibration
            cal_key = self._get_broadcast_key(station, freq_mhz)
            cal = self.calibration.get(cal_key)
            if cal and cal.n_samples > 10:
                d_clock_calibrated = d_clock_raw + cal.offset_ms
            else:
                d_clock_calibrated = d_clock_raw
            
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
            
            weight = max(0.01, confidence * grade_w * mode_w * snr_w)
            
            measurements.append((broadcast_key, d_clock_raw, d_clock_calibrated, weight, station, freq_mhz))
            
            # Update calibration history
            if cal_key not in self.calibration_history:
                self.calibration_history[cal_key] = deque(maxlen=self.calibration_history_max)
            self.calibration_history[cal_key].append(d_clock_raw)
        
        if len(measurements) < 2:
            return None
        
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
            
            # Reject outliers > 3 MAD
            keep_mask = deviations < (3.0 * mad)
            
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
        
        return FusionResult(
            d_clock_ms=fused_d_clock,
            d_clock_raw_ms=raw_mean,
            uncertainty_ms=uncertainty,
            n_broadcasts=n_broadcasts,
            n_outliers_rejected=n_rejected,
            quality_grade=grade,
            wwv_count=wwv_count,
            wwvh_count=wwvh_count,
            chu_count=chu_count
        )
    
    def _update_calibration(self, measurements: List):
        """
        Update per-broadcast calibration using EMA.
        
        Calibration offset brings mean D_clock toward 0 (UTC alignment).
        Uses exponential moving average with α = max(0.5, 20/n_samples).
        """
        for broadcast_key, d_clock_raw, _, _, station, freq_mhz in measurements:
            cal_key = self._get_broadcast_key(station, freq_mhz)
            history = self.calibration_history.get(cal_key)
            
            if not history or len(history) < 10:
                continue
            
            # Recent measurements
            recent = list(history)[-30:]
            mean_d_clock = np.mean(recent)
            std_d_clock = np.std(recent) if len(recent) > 1 else 1.0
            
            # New offset should bring mean to 0
            new_offset = -mean_d_clock
            
            # EMA update
            old_cal = self.calibration.get(cal_key)
            if old_cal and old_cal.n_samples > 0:
                alpha = max(0.5, 20.0 / old_cal.n_samples)
                new_offset = alpha * new_offset + (1 - alpha) * old_cal.offset_ms
            
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
        """Update Chrony SHM with current D_clock."""
        if not self.chrony_shm:
            return
        
        try:
            # Convert D_clock (ms) to offset (seconds)
            offset = self.d_clock_ms / 1000.0
            self.chrony_shm.update(offset, time.time())
            self.stats['chrony_updates'] += 1
            logger.debug(f"Chrony updated: offset={offset*1000:.2f}ms")
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
        
        # Initialize and start RTP receiver
        self._init_rtp_receiver()
        self.rtp_receiver.start()
        
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
        logger.info(f"  Slow Loop: waiting for T=:02")
    
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
