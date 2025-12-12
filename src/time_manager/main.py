#!/usr/bin/env python3
"""
time-manager: Precision HF Time Transfer Daemon

Main entry point for the time-manager daemon. This service:
1. Reads IQ data from radiod channels (WWV/WWVH/CHU)
2. Extracts timing using tone detection and BCD correlation
3. Computes D_clock (system clock offset from UTC)
4. Publishes results to shared memory for consumers
5. Optionally feeds chronyd to discipline system clock

Usage:
    # Start daemon
    python -m time_manager --config /etc/time-manager/config.toml
    
    # Single channel test mode
    python -m time_manager --test-channel "WWV 10 MHz" --data-root /tmp/grape-test

Architecture:
    This daemon is INFRASTRUCTURE, not a science application.
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                        time-manager                              │
    │                                                                  │
    │  ┌──────────┐   ┌──────────────┐   ┌────────────────────────┐   │
    │  │  radiod  │──▶│ Per-Channel  │──▶│ Multi-Broadcast Fusion │   │
    │  │ (9 ch)   │   │  Processing  │   │                        │   │
    │  └──────────┘   └──────────────┘   └────────────────────────┘   │
    │                                              │                   │
    │                        ┌─────────────────────┴──────────────┐   │
    │                        ▼                                    ▼   │
    │               /dev/shm/grape_timing              Chrony SHM     │
    │               (for grape-recorder)               (for OS)       │
    └─────────────────────────────────────────────────────────────────┘

Consumers:
    - grape-recorder: Reads /dev/shm/grape_timing for D_clock and station ID
    - chronyd: Reads SHM refclock for system clock discipline
    - Other apps: Can read SHM for timing without implementing WWV logic
"""

import argparse
import json
import logging
import signal
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, List, Any
import toml

# Set up logging before imports that use it
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('time-manager')

from .interfaces.timing_result import (
    TimingResult,
    ChannelTimingResult,
    FusionResult,
    DiscriminationInfo,
    ClockStatus,
)
from .output.shm_writer import SHMWriter
from .output.chrony_shm import ChronySHM


def channel_name_to_dir(channel_name: str) -> str:
    """
    Convert channel name to directory-safe format.
    
    Examples:
        "WWV 10 MHz" -> "WWV_10_MHz"
        "CHU 3.33 MHz" -> "CHU_3.33_MHz"
    
    Args:
        channel_name: Human-readable channel name with spaces
        
    Returns:
        Directory-safe name with underscores
    """
    return channel_name.replace(' ', '_')


class TimeManagerDaemon:
    """
    Main time-manager daemon.
    
    Orchestrates timing extraction from WWV/WWVH/CHU broadcasts and
    publishes results to shared memory.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        data_root: Optional[Path] = None
    ):
        """
        Initialize the time-manager daemon.
        
        Args:
            config: Configuration dictionary
            data_root: Override data root path (for testing)
        """
        self.config = config
        # data_root can be at top level or under [general]
        default_data_root = config.get('general', {}).get('data_root', 
                           config.get('data_root', '/tmp/grape-test'))
        self.data_root = Path(data_root or default_data_root)
        
        # Channels to process
        # Config can have channels as list or as [channels].enabled
        channels_config = config.get('channels', [])
        if isinstance(channels_config, dict):
            self.channels = channels_config.get('enabled', [])
        elif isinstance(channels_config, list):
            self.channels = channels_config
        else:
            self.channels = []
        
        # Default channels if none configured
        if not self.channels:
            self.channels = [
                'WWV 2.5 MHz', 'WWV 5 MHz', 'WWV 10 MHz', 'WWV 15 MHz',
                'WWV 20 MHz', 'WWV 25 MHz',
                'CHU 3.33 MHz', 'CHU 7.85 MHz', 'CHU 14.67 MHz'
            ]
        
        # Receiver configuration
        self.receiver_grid = config.get('receiver', {}).get('grid_square', 'EM38ww')
        self.receiver_lat = config.get('receiver', {}).get('latitude')
        self.receiver_lon = config.get('receiver', {}).get('longitude')
        # sample_rate can be at top level or under [general]
        self.sample_rate = config.get('general', {}).get('sample_rate',
                          config.get('sample_rate', 20000))
        
        # Output configuration
        self.shm_path = config.get('output', {}).get('shm_path', '/dev/shm/grape_timing')
        self.enable_chrony = config.get('output', {}).get('enable_chrony', False)
        self.chrony_unit = config.get('output', {}).get('chrony_unit', 0)
        
        # Initialize outputs
        self.shm_writer = SHMWriter(self.shm_path)
        self.chrony_shm: Optional[ChronySHM] = None
        if self.enable_chrony:
            self.chrony_shm = ChronySHM(unit=self.chrony_unit)
        
        # State
        self.running = False
        self.start_time = 0.0
        self.minutes_processed = 0
        self.last_result: Optional[TimingResult] = None
        
        # Per-channel engines (will be populated on start)
        self.channel_engines: Dict[str, Any] = {}
        
        # Clock state
        self.clock_status = ClockStatus.ACQUIRING
        self.d_clock_ms = 0.0
        self.d_clock_uncertainty_ms = 10.0
        
        logger.info("=" * 60)
        logger.info("time-manager initializing")
        logger.info(f"  Data root: {self.data_root}")
        logger.info(f"  Channels: {len(self.channels)}")
        logger.info(f"  Receiver: {self.receiver_grid}")
        logger.info(f"  SHM output: {self.shm_path}")
        logger.info(f"  Chrony enabled: {self.enable_chrony}")
        logger.info("=" * 60)
    
    def start(self):
        """Start the daemon."""
        logger.info("Starting time-manager daemon")
        
        self.running = True
        self.start_time = time.time()
        
        # Set up signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Connect to Chrony SHM if enabled
        if self.chrony_shm:
            if self.chrony_shm.connect():
                logger.info("Connected to Chrony SHM")
            else:
                logger.warning("Failed to connect to Chrony SHM - continuing without")
                self.chrony_shm = None
        
        # Initialize per-channel engines
        self._init_channel_engines()
        
        # Main processing loop
        try:
            self._main_loop()
        except Exception as e:
            logger.exception(f"Fatal error in main loop: {e}")
            raise
        finally:
            self._cleanup()
    
    def _init_channel_engines(self):
        """Initialize timing engines for each channel."""
        logger.info("Initializing channel engines...")
        
        # Import from local timing module
        try:
            from .timing import Phase2TemporalEngine
            
            for channel_name in self.channels:
                channel_dir = channel_name_to_dir(channel_name)
                raw_buffer_dir = self.data_root / 'raw_buffer' / channel_dir
                
                if not raw_buffer_dir.exists():
                    logger.warning(f"No raw_buffer for {channel_name}, skipping")
                    continue
                
                # Extract frequency from channel name
                parts = channel_name.split()
                freq_mhz = float(parts[1])
                frequency_hz = freq_mhz * 1e6
                
                engine = Phase2TemporalEngine(
                    raw_archive_dir=self.data_root / 'raw_buffer',
                    output_dir=self.data_root / 'phase2' / channel_dir,
                    channel_name=channel_name,
                    frequency_hz=frequency_hz,
                    receiver_grid=self.receiver_grid,
                    sample_rate=self.sample_rate,
                    precise_lat=self.receiver_lat,
                    precise_lon=self.receiver_lon
                )
                
                self.channel_engines[channel_name] = engine
                logger.info(f"  Initialized: {channel_name}")
            
            logger.info(f"Initialized {len(self.channel_engines)} channel engines")
            
        except ImportError as e:
            logger.error(f"Failed to import timing modules: {e}")
            logger.error("Using stub engines for testing")
            
            # Create stub engines for testing
            for channel_name in self.channels:
                self.channel_engines[channel_name] = None
    
    def _main_loop(self):
        """Main processing loop."""
        logger.info("Entering main loop")
        
        poll_interval = self.config.get('poll_interval', 10.0)
        last_minute_processed: Dict[str, int] = {}
        
        while self.running:
            try:
                current_minute = int(time.time() / 60)
                channel_results: Dict[str, ChannelTimingResult] = {}
                
                # Process each channel
                for channel_name, engine in self.channel_engines.items():
                    if engine is None:
                        continue
                    
                    # Check if we have new data to process
                    last_min = last_minute_processed.get(channel_name, 0)
                    
                    # Try to process previous minute (current minute may not be complete)
                    target_minute = current_minute - 1
                    
                    if target_minute <= last_min:
                        continue
                    
                    result = self._process_channel_minute(
                        channel_name, engine, target_minute
                    )
                    
                    if result:
                        channel_results[channel_name] = result
                        last_minute_processed[channel_name] = target_minute
                
                # Fuse results from all channels
                if channel_results:
                    self._fuse_and_publish(channel_results)
                    self.minutes_processed += 1
                
                # Sleep until next poll
                time.sleep(poll_interval)
                
            except Exception as e:
                logger.exception(f"Error in main loop iteration: {e}")
                time.sleep(1)
    
    def _process_channel_minute(
        self,
        channel_name: str,
        engine: Any,
        minute: int
    ) -> Optional[ChannelTimingResult]:
        """
        Process one minute of data for a channel.
        
        Args:
            channel_name: Channel name
            engine: Phase2TemporalEngine instance
            minute: Unix minute number to process
            
        Returns:
            ChannelTimingResult or None if no data
        """
        try:
            # Find the data file for this minute
            channel_dir = channel_name_to_dir(channel_name)
            minute_ts = minute * 60
            dt = datetime.fromtimestamp(minute_ts, tz=timezone.utc)
            date_str = dt.strftime('%Y%m%d')
            
            # Look for binary minute file
            raw_buffer_dir = self.data_root / 'raw_buffer' / channel_dir / date_str
            if not raw_buffer_dir.exists():
                return None
            
            # Find the file for this minute
            # Files are named {unix_timestamp}.bin (e.g., 1765338300.bin)
            minute_file = raw_buffer_dir / f"{minute_ts}.bin"
            
            if not minute_file.exists():
                logger.debug(f"{channel_name}: File not found: {minute_file}")
                return None
            
            # Load the binary file
            import numpy as np
            
            # Read metadata
            meta_file = minute_file.with_suffix('.json')
            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                rtp_timestamp = metadata.get('start_rtp_timestamp', minute_ts * self.sample_rate)
            else:
                rtp_timestamp = minute_ts * self.sample_rate
            
            # Read IQ samples
            iq_samples = np.fromfile(minute_file, dtype=np.complex64)
            
            if len(iq_samples) < self.sample_rate * 30:  # Need at least 30 seconds
                logger.debug(f"{channel_name}: Insufficient samples ({len(iq_samples)})")
                return None
            
            # Process through the engine
            result = engine.process_minute(
                iq_samples=iq_samples,
                system_time=float(minute_ts),
                rtp_timestamp=rtp_timestamp
            )
            
            if result is None:
                return None
            
            # Convert to ChannelTimingResult
            is_shared = self._is_shared_frequency(channel_name)
            
            disc_info = None
            if is_shared and result.channel:
                disc_info = DiscriminationInfo(
                    method=result.channel.ground_truth_source or 'power_ratio',
                    power_ratio_db=getattr(result.time_snap, 'wwv_snr_db', None),
                    differential_delay_ms=result.channel.bcd_differential_delay_ms,
                    bcd_correlation_quality=result.channel.bcd_correlation_quality,
                    confidence_score=result.channel.station_confidence == 'high' and 0.9 or 0.5
                )
            
            return ChannelTimingResult(
                channel_name=channel_name,
                station=result.solution.station if result.solution else 'UNKNOWN',
                confidence=result.channel.station_confidence if result.channel else 'low',
                tone_detected=result.time_snap.wwv_detected or result.time_snap.wwvh_detected,
                timing_error_ms=result.time_snap.timing_error_ms,
                propagation_delay_ms=result.solution.t_propagation_ms if result.solution else None,
                d_clock_raw_ms=result.d_clock_ms,
                propagation_mode=result.solution.propagation_mode if result.solution else 'UNKNOWN',
                n_hops=result.solution.n_hops if result.solution else 0,
                layer_height_km=result.solution.layer_height_km if result.solution else 0.0,
                snr_db=result.time_snap.wwv_snr_db or result.time_snap.wwvh_snr_db,
                discrimination=disc_info,
                is_shared_frequency=is_shared,
                uncertainty_ms=result.uncertainty_ms
            )
            
        except Exception as e:
            logger.debug(f"{channel_name}: Processing error: {e}")
            return None
    
    def _is_shared_frequency(self, channel_name: str) -> bool:
        """Check if channel is on a shared WWV/WWVH frequency."""
        shared_freqs = ['2.5', '5', '10', '15']
        for freq in shared_freqs:
            if f"{freq} MHz" in channel_name:
                return True
        return False
    
    def _fuse_and_publish(self, channel_results: Dict[str, ChannelTimingResult]):
        """
        Fuse results from all channels and publish to SHM.
        
        Args:
            channel_results: Per-channel timing results
        """
        # Simple fusion: weighted average of D_clock values
        total_weight = 0.0
        weighted_sum = 0.0
        contributing = 0
        
        for ch_name, ch_result in channel_results.items():
            if ch_result.d_clock_raw_ms is not None:
                # Weight by inverse uncertainty squared
                weight = 1.0 / (ch_result.uncertainty_ms ** 2)
                weighted_sum += weight * ch_result.d_clock_raw_ms
                total_weight += weight
                contributing += 1
        
        if total_weight > 0:
            fused_d_clock = weighted_sum / total_weight
            fused_uncertainty = 1.0 / (total_weight ** 0.5)
            
            self.d_clock_ms = fused_d_clock
            self.d_clock_uncertainty_ms = fused_uncertainty
            
            # Update clock status
            if contributing >= 3 and fused_uncertainty < 2.0:
                self.clock_status = ClockStatus.LOCKED
            elif contributing >= 1:
                self.clock_status = ClockStatus.ACQUIRING
            else:
                self.clock_status = ClockStatus.UNLOCKED
        
        # Create fusion result
        fusion = FusionResult(
            contributing_broadcasts=contributing,
            total_broadcasts=13,
            fused_d_clock_ms=self.d_clock_ms,
            fusion_uncertainty_ms=self.d_clock_uncertainty_ms
        )
        
        # Create and publish timing result
        now = time.time()
        result = TimingResult(
            timestamp=now,
            system_time=now,
            d_clock_ms=self.d_clock_ms,
            d_clock_uncertainty_ms=self.d_clock_uncertainty_ms,
            clock_status=self.clock_status,
            fusion=fusion,
            channels={name: ch for name, ch in channel_results.items()},
            channels_active=len(channel_results),
            channels_locked=contributing,
            uptime_seconds=now - self.start_time
        )
        
        # Write to SHM for consumers
        self.shm_writer.write(result)
        self.last_result = result
        
        # Update Chrony if enabled and locked
        if self.chrony_shm and self.clock_status == ClockStatus.LOCKED:
            # Reference time is system time minus D_clock
            reference_time = now - (self.d_clock_ms / 1000.0)
            self.chrony_shm.update(
                reference_time=reference_time,
                system_time=now,
                precision=-10  # ~1ms
            )
        
        logger.info(
            f"Fused: d_clock={self.d_clock_ms:+.2f}ms ±{self.d_clock_uncertainty_ms:.2f}ms, "
            f"status={self.clock_status.value}, channels={contributing}/{len(channel_results)}"
        )
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def _cleanup(self):
        """Clean up resources on shutdown."""
        logger.info("Cleaning up...")
        
        if self.chrony_shm:
            self.chrony_shm.disconnect()
        
        logger.info(f"Processed {self.minutes_processed} minutes")
        logger.info("time-manager stopped")
    
    def reprocess(
        self,
        num_minutes: int = 10,
        channel_filter: Optional[str] = None
    ):
        """
        Reprocess historical data instead of polling for new data.
        
        This mode discovers existing minute files and processes them,
        useful for testing and batch reprocessing.
        
        Args:
            num_minutes: Maximum number of minutes to process
            channel_filter: If set, only process this channel
        """
        logger.info("=" * 60)
        logger.info("REPROCESS MODE")
        logger.info(f"  Data root: {self.data_root}")
        logger.info(f"  Max minutes: {num_minutes}")
        logger.info(f"  Channel filter: {channel_filter or 'all'}")
        logger.info("=" * 60)
        
        self.running = True
        self.start_time = time.time()
        
        # Initialize engines
        self._init_channel_engines()
        
        # Filter channels if requested
        if channel_filter:
            if channel_filter not in self.channel_engines:
                logger.error(f"Channel '{channel_filter}' not found. Available: {list(self.channel_engines.keys())}")
                return
            engines_to_process = {channel_filter: self.channel_engines[channel_filter]}
        else:
            engines_to_process = self.channel_engines
        
        # Discover available minute files
        all_minutes: Dict[str, List[int]] = {}
        
        for channel_name in engines_to_process.keys():
            channel_dir = channel_name_to_dir(channel_name)
            raw_buffer_path = self.data_root / 'raw_buffer' / channel_dir
            
            logger.info(f"Scanning {raw_buffer_path}")
            
            if not raw_buffer_path.exists():
                logger.warning(f"  No raw_buffer for {channel_name}")
                continue
            
            minutes = []
            # Scan all date directories
            for date_dir in sorted(raw_buffer_path.iterdir()):
                if not date_dir.is_dir():
                    continue
                # Find all .bin files
                for bin_file in date_dir.glob("*.bin"):
                    try:
                        minute_ts = int(bin_file.stem)
                        minutes.append(minute_ts // 60)  # Convert to minute number
                    except ValueError:
                        continue
            
            minutes = sorted(set(minutes))
            all_minutes[channel_name] = minutes
            logger.info(f"  Found {len(minutes)} minutes for {channel_name}")
            if minutes:
                logger.info(f"    Range: {minutes[0]} to {minutes[-1]}")
        
        if not all_minutes:
            logger.error("No data found to reprocess")
            return
        
        # Find common minutes across all channels for proper fusion
        # Get union of all minutes, then filter to most recent N
        all_minute_set = set()
        for minutes in all_minutes.values():
            all_minute_set.update(minutes)
        
        sorted_minutes = sorted(all_minute_set)[-num_minutes:]
        logger.info(f"\nProcessing {len(sorted_minutes)} minutes with multi-channel fusion")
        
        processed_count = 0
        
        for minute in sorted_minutes:
            channel_results: Dict[str, ChannelTimingResult] = {}
            
            # Process all channels for this minute
            for channel_name, engine in engines_to_process.items():
                if engine is None:
                    continue
                
                # Check if this channel has data for this minute
                if minute not in all_minutes.get(channel_name, []):
                    continue
                
                result = self._process_channel_minute(channel_name, engine, minute)
                
                if result:
                    channel_results[channel_name] = result
            
            # Log results for this minute
            if channel_results:
                processed_count += 1
                
                # Compute fused D_clock
                self._fuse_and_publish(channel_results)
                
                # Log summary
                d_clock_values = [
                    f"{ch[:8]}={r.d_clock_raw_ms:+.1f}" 
                    for ch, r in sorted(channel_results.items()) 
                    if r.d_clock_raw_ms is not None
                ]
                logger.info(
                    f"  Minute {minute}: {len(channel_results)} channels, "
                    f"fused={self.d_clock_ms:+.2f}±{self.d_clock_uncertainty_ms:.2f}ms"
                )
                if len(d_clock_values) <= 5:
                    logger.info(f"    {', '.join(d_clock_values)}")
            else:
                logger.debug(f"  Minute {minute}: no results")
        
        logger.info("=" * 60)
        logger.info(f"Reprocessing complete: {processed_count} fused minutes")
        logger.info(f"Final D_clock: {self.d_clock_ms:+.2f} ± {self.d_clock_uncertainty_ms:.2f} ms")
        logger.info(f"Clock status: {self.clock_status.value}")
        logger.info("=" * 60)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from TOML file."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return toml.load(f)
    
    # Default configuration
    return {
        'data_root': '/tmp/grape-test',
        'sample_rate': 20000,
        'poll_interval': 10.0,
        'channels': [
            'WWV 2.5 MHz', 'WWV 5 MHz', 'WWV 10 MHz', 'WWV 15 MHz',
            'WWV 20 MHz', 'WWV 25 MHz',
            'CHU 3.33 MHz', 'CHU 7.85 MHz', 'CHU 14.67 MHz'
        ],
        'receiver': {
            'grid_square': 'EM38ww',
            'latitude': 38.918461,
            'longitude': -92.127974
        },
        'output': {
            'shm_path': '/dev/shm/grape_timing',
            'enable_chrony': False,
            'chrony_unit': 0
        }
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='time-manager: Precision HF Time Transfer Daemon',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start with config file
    python -m time_manager --config /etc/time-manager/config.toml
    
    # Test mode with specific data root
    python -m time_manager --data-root /tmp/grape-test
    
    # Enable Chrony SHM integration
    python -m time_manager --enable-chrony
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Path to TOML configuration file'
    )
    parser.add_argument(
        '--data-root', '-d',
        help='Data root directory (overrides config)'
    )
    parser.add_argument(
        '--enable-chrony',
        action='store_true',
        help='Enable Chrony SHM integration'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--reprocess',
        action='store_true',
        help='Reprocess historical data instead of polling for new data'
    )
    parser.add_argument(
        '--minutes', '-n',
        type=int,
        default=10,
        help='Number of minutes to process in reprocess mode (default: 10)'
    )
    parser.add_argument(
        '--channel',
        help='Process only this channel (e.g., "WWV 10 MHz")'
    )
    parser.add_argument(
        '--live',
        action='store_true',
        help='Run in live mode (subscribe to RTP multicast, Twin-Stream architecture)'
    )
    parser.add_argument(
        '--multicast',
        default='239.1.2.82',
        help='RTP multicast address (default: 239.1.2.82)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5004,
        help='RTP port (default: 5004)'
    )
    parser.add_argument(
        '--status-addr',
        default='radiod.local',
        help='Radiod status multicast address for channel discovery (default: radiod.local)'
    )
    parser.add_argument(
        '--health-port',
        type=int,
        default=8080,
        help='HTTP port for health monitoring endpoint (default: 8080, 0 to disable)'
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply command-line overrides
    if args.data_root:
        config['data_root'] = args.data_root
    if args.enable_chrony:
        config.setdefault('output', {})['enable_chrony'] = True
    
    if args.live:
        # Live mode: Twin-Stream architecture with RTP subscription
        from .engine.live_time_engine import LiveTimeEngine
        
        # Build channel list - auto-discover if SSRCs not configured
        channels = []
        channels_config = config.get('channels', {})
        if isinstance(channels_config, dict):
            channel_names = channels_config.get('enabled', [])
            ssrc_map = channels_config.get('ssrc', {})
        else:
            channel_names = channels_config if isinstance(channels_config, list) else []
            ssrc_map = {}
        
        # Only use channels with configured SSRCs
        for name in channel_names:
            ssrc = ssrc_map.get(name, 0)
            if ssrc:
                # Extract frequency from channel name
                parts = name.split()
                freq_mhz = float(parts[1])
                channels.append({
                    'name': name,
                    'ssrc': ssrc,
                    'frequency_hz': freq_mhz * 1e6
                })
        
        # If no SSRCs configured, engine will auto-discover from radiod
        if not channels:
            logger.info("No SSRCs configured - will auto-discover from radiod")
        
        # Get receiver config
        receiver_config = config.get('receiver', {})
        
        # Get RTP config from file or CLI
        rtp_config = config.get('rtp', {})
        multicast_addr = rtp_config.get('multicast_address', args.multicast)
        rtp_port = rtp_config.get('port', args.port)
        status_addr = rtp_config.get('status_address')
        
        # If no status_address configured, discover available radiod instances
        if not status_addr:
            try:
                from ka9q.discovery import discover_radiod_services
                logger.info("No status_address configured, discovering radiod instances...")
                services = discover_radiod_services(timeout=3.0)
                
                if not services:
                    logger.error("No radiod instances found on the network.")
                    logger.error("Please configure 'status_address' in [rtp] section of config file.")
                    sys.exit(1)
                elif len(services) == 1:
                    status_addr = services[0]
                    logger.info(f"Found radiod: {status_addr}")
                else:
                    print("\nAvailable radiod instances:")
                    for i, svc in enumerate(services, 1):
                        print(f"  {i}. {svc}")
                    print()
                    while True:
                        try:
                            choice = input(f"Select radiod instance [1-{len(services)}]: ").strip()
                            idx = int(choice) - 1
                            if 0 <= idx < len(services):
                                status_addr = services[idx]
                                break
                            print(f"Please enter a number between 1 and {len(services)}")
                        except ValueError:
                            print("Please enter a valid number")
                        except (KeyboardInterrupt, EOFError):
                            print("\nAborted.")
                            sys.exit(1)
                    logger.info(f"Selected radiod: {status_addr}")
            except ImportError:
                logger.error("ka9q-python not installed. Please install it or configure status_address manually.")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Failed to discover radiod: {e}")
                logger.error("Please configure 'status_address' in [rtp] section of config file.")
                sys.exit(1)
        
        engine = LiveTimeEngine(
            multicast_address=multicast_addr,
            port=rtp_port,
            channels=channels,
            sample_rate=config.get('general', {}).get('sample_rate', 20000),
            receiver_grid=receiver_config.get('grid_square', 'EM38ww'),
            receiver_lat=receiver_config.get('latitude'),
            receiver_lon=receiver_config.get('longitude'),
            enable_chrony=args.enable_chrony or config.get('output', {}).get('enable_chrony', False),
            chrony_unit=config.get('output', {}).get('chrony_unit', 0),
            status_address=status_addr
        )
        
        # Start web server (includes health endpoints + GUI)
        web_server = None
        web_port = config.get('output', {}).get('health_port', args.health_port)
        if web_port > 0:
            from .web import WebServer
            web_server = WebServer(port=web_port)
            web_server.set_engine(engine)
            web_server.start()
        
        try:
            engine.run()
        finally:
            if web_server:
                web_server.stop()
    
    elif args.reprocess:
        # Reprocess mode: Read from disk files
        daemon = TimeManagerDaemon(config, data_root=args.data_root)
        daemon.reprocess(
            num_minutes=args.minutes,
            channel_filter=args.channel
        )
    
    else:
        # Daemon mode: Poll for new disk files
        daemon = TimeManagerDaemon(config, data_root=args.data_root)
        daemon.start()


if __name__ == '__main__':
    main()
