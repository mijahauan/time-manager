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
            
            # Use wallclock or estimate from RTP
            wc = wallclock or (header.timestamp / self.sample_rate)
            
            # Add to buffer
            self.channel_buffers[channel_name].add_samples(
                samples, header.timestamp, wc
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
        """Initialize RTP receiver with callbacks for all channels."""
        # Import from grape_recorder core
        import sys
        sys.path.insert(0, str(Path.home() / 'grape-recorder' / 'src'))
        from grape_recorder.core.rtp_receiver import RTPReceiver
        
        # If no channels configured, try to discover them
        if not self.channels:
            logger.info("No channels configured, discovering from radiod...")
            self.channels = self._discover_channels(self.status_address)
            
            if not self.channels:
                logger.error("No channels discovered - is radiod running?")
                logger.error("Check: systemctl status radiod")
                raise RuntimeError("No channels discovered from radiod")
            
            # Re-initialize buffers for discovered channels
            self._init_channel_buffers()
        
        self.rtp_receiver = RTPReceiver(self.multicast_address, self.port)
        
        # Register callback for each channel
        for ch_config in self.channels:
            name = ch_config['name']
            ssrc = ch_config['ssrc']
            channel_info = ch_config.get('channel_info')
            
            self.rtp_receiver.register_callback(
                ssrc=ssrc,
                callback=self._rtp_callback(name),
                channel_info=channel_info  # Enable wallclock timing
            )
            logger.info(f"  Registered RTP callback: {name} (SSRC={ssrc})")
    
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
        
        # Skip if still acquiring
        if self.state == EngineState.ACQUIRING:
            logger.info("  Skipping (still acquiring)")
            return
        
        d_clock_values = []
        
        for name, buffer in self.channel_buffers.items():
            state = self.channel_states.get(name)
            
            if not state or not state.is_valid():
                logger.debug(f"  {name}: No valid state")
                continue
            
            # Get ring buffer samples
            samples, start_rtp = buffer.get_ring_samples()
            
            if len(samples) < self.sample_rate:  # Need at least 1 second
                logger.debug(f"  {name}: Insufficient samples ({len(samples)})")
                continue
            
            # TODO: Run tone detection to find minute boundary
            # For now, use state's propagation delay directly
            # tone_time = self._detect_tone_fast(samples, start_rtp, name)
            
            # Calculate D_clock using previous state
            # D_clock = Arrival_RTP - Propagation_Delay
            # This is simplified - full implementation needs tone detection
            d_clock = 0.0  # Placeholder
            
            if state.propagation_delay_ms > 0:
                d_clock_values.append((d_clock, state.confidence))
                logger.debug(f"  {name}: D_clock={d_clock:.2f}ms (using {state.station} {state.propagation_mode})")
        
        if d_clock_values:
            # Weighted average
            total_weight = sum(conf for _, conf in d_clock_values)
            if total_weight > 0:
                self.d_clock_ms = sum(d * c for d, c in d_clock_values) / total_weight
            
            # Update Chrony if enabled
            if self.enable_chrony and self.chrony_shm:
                self._update_chrony()
            
            logger.info(f"  Fast Loop result: D_clock={self.d_clock_ms:.2f}ms")
    
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
        
        for name, buffer in self.channel_buffers.items():
            # Get full minute samples
            samples, start_rtp = buffer.get_full_samples()
            
            if len(samples) < self.sample_rate * 30:  # Need at least 30 seconds
                logger.debug(f"  {name}: Insufficient samples ({len(samples)})")
                continue
            
            # TODO: Run full discrimination
            # - BCD correlation
            # - Doppler analysis
            # - FSS calculation
            # - Station identification
            
            # For now, use placeholder results
            result = {
                'station': 'WWV',
                'propagation_mode': '1F',
                'propagation_delay_ms': 6.0,
                'snr_db': 20.0,
                'confidence': 0.8
            }
            
            results[name] = result
            
            # Update channel state for next Fast Loop
            state = self.channel_states[name]
            state.station = result['station']
            state.propagation_mode = result['propagation_mode']
            state.propagation_delay_ms = result['propagation_delay_ms']
            state.snr_db = result['snr_db']
            state.confidence = result['confidence']
            state.last_update_minute = minute
            
            logger.info(f"  {name}: {result['station']} {result['propagation_mode']} "
                       f"delay={result['propagation_delay_ms']:.1f}ms")
            
            # Reset buffer for next minute
            buffer.reset_for_new_minute()
        
        # Transition from ACQUIRING to TRACKING
        if self.state == EngineState.ACQUIRING and results:
            self.state = EngineState.TRACKING
            logger.info("State transition: ACQUIRING -> TRACKING")
        
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
        
        # Stop RTP receiver
        if self.rtp_receiver:
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
