#!/usr/bin/env python3
"""
Phase 2: Temporal Analysis Engine - Precision Timing Analytics

================================================================================
PURPOSE
================================================================================
The Phase 2 Temporal Engine is the CENTRAL ORCHESTRATOR for all timing analytics.
It coordinates the three-step process that transforms raw IQ samples into a
precision D_clock measurement:

    D_clock = T_system - T_UTC(NIST)

This is the "System Clock Offset" - the primary output of the GRAPE system.

================================================================================
ARCHITECTURAL OVERVIEW
================================================================================
Phase 2 implements a hierarchical refinement strategy where each step narrows
the search window for the next:

┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 1: TIME SNAP (±500ms → anchor)                      │
│                                                                             │
│   Input:  Raw IQ @ 20 kHz, system_time, rtp_timestamp                       │
│   Method: Matched filter tone detection (1000/1200 Hz)                      │
│   Output: timing_error_ms, anchor_station, confidence                       │
│                                                                             │
│   🎯 Establishes initial temporal synchronization                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│             STEP 2: CHANNEL CHARACTERIZATION (±50ms window)                 │
│                                                                             │
│   2A. BCD Correlation → differential_delay_ms, dual-peak timing             │
│   2B. Doppler Estimation → doppler_std_hz, coherence_time                   │
│   2C. Station Discrimination → dominant_station, ground_truth               │
│   2D. Test Signal Analysis → FSS, delay_spread (minutes 8/44)               │
│                                                                             │
│   📡 Characterizes ionospheric channel for mode disambiguation              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                STEP 3: TRANSMISSION TIME SOLUTION (→ D_clock)               │
│                                                                             │
│   Input:  timing_error_ms + channel_metrics + station_ID                    │
│   Method: TransmissionTimeSolver with mode disambiguation                   │
│   Output: D_clock, propagation_mode, confidence, uncertainty                │
│                                                                             │
│   🎯 Back-calculates UTC(NIST) from observed arrival time                   │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
DATA FLOW: THE D_CLOCK EQUATION
================================================================================
The fundamental equation we're solving:

    T_arrival = T_emission + T_propagation + D_clock

Where:
    T_arrival = Observed tone arrival time (from Step 1)
    T_emission = 0 (tones transmitted at exact second boundary)
    T_propagation = HF signal propagation delay (from Step 3 mode solving)
    D_clock = System clock offset (THE OUTPUT WE WANT)

Rearranging:
    D_clock = T_arrival - T_propagation

================================================================================
STEP 1: TIME SNAP - INITIAL SYNCHRONIZATION
================================================================================
The tone detector (from tone_detector.py) uses quadrature matched filtering
to detect the 800ms timing tones:

    - WWV:  1000 Hz, 0.8s duration at second 0
    - WWVH: 1200 Hz, 0.8s duration at second 0
    - CHU:  1000 Hz, 0.5s duration at second 0 (1.0s at hour)

Output: timing_error_ms = offset from expected minute boundary
        This is typically in the range of 5-50ms (propagation delay)

SEARCH WINDOW: ±500ms (wide, to handle unknown propagation)

================================================================================
STEP 2: CHANNEL CHARACTERIZATION
================================================================================
With timing anchored to ±50ms, Step 2 extracts channel metrics:

STEP 2A: BCD CORRELATION
    The 100 Hz Binary Coded Decimal subcarrier provides:
    - Differential delay between WWV and WWVH peaks
    - Amplitude ratio for station power comparison
    - Delay spread from correlation peak width

STEP 2B: DOPPLER ESTIMATION  
    Per-tick phase tracking measures:
    - Doppler shift (ionospheric motion)
    - Doppler standard deviation (channel stability)
    - Maximum coherent integration window

STEP 2C: STATION DISCRIMINATION
    Weighted voting across multiple methods (see wwvh_discrimination.py):
    - Ground truth tones (500/600 Hz, 440 Hz)
    - Power ratio (1000 Hz vs 1200 Hz)
    - BCD amplitude ratio
    - Test signal (minutes 8/44)

STEP 2D: TEST SIGNAL ANALYSIS (Minutes 8 and 44 only)
    Scientific modulation test provides:
    - Frequency Selectivity Score (FSS) - D-layer indicator for mode disambiguation
    - Delay spread from chirp analysis - multipath severity
    - High-precision ToA from single-cycle bursts
    - Coherence time from fading analysis

================================================================================
STEP 3: TRANSMISSION TIME SOLUTION
================================================================================
The TransmissionTimeSolver (from transmission_time_solver.py) identifies the
propagation mode and computes D_clock:

STATION PRIORITY FOR MODE SOLVING:
    1. Ground truth (500/600 Hz exclusive minutes)
    2. High-confidence discrimination
    3. Channel name (e.g., "WWV 20 MHz" is unambiguous)
    4. Fallback to WWV

MODE DISAMBIGUATION INPUTS:
    - delay_spread_ms: High → favor multi-hop modes
    - doppler_std_hz: High → unstable path, reduce confidence
    - fss_db: Negative → D-layer attenuation, favor multi-hop

OUTPUT:
    - d_clock_ms: System clock offset from UTC(NIST)
    - propagation_mode: '1F', '2F', 'GW', etc.
    - confidence: 0-1 confidence in the solution
    - uncertainty_ms: Estimated timing uncertainty

================================================================================
INPUT DATA REQUIREMENTS
================================================================================
- Data format: np.complex64 (32-bit float I + 32-bit float Q)
- Sample rate: 20,000 Hz (full Phase 1 resolution)
- Buffer duration: 60 seconds (one complete minute)
- Source: Phase 1 Digital RF archive (IMMUTABLE - never modified)

32-BIT FLOAT RATIONALE:
    - 144 dB dynamic range vs 96 dB for 16-bit
    - AGC disabled (F32 has sufficient range)
    - Preserves weak signal information
    - Consistent amplitude for matched filtering

================================================================================
USAGE
================================================================================
    from .phase2_temporal_engine import Phase2TemporalEngine
    
    engine = Phase2TemporalEngine(
        raw_archive_dir=Path('/data/raw_archive'),
        output_dir=Path('/data/phase2'),
        channel_name='WWV_10MHz',
        frequency_hz=10e6,
        receiver_grid='EM38ww'
    )
    
    # Process a minute of data
    result = engine.process_minute(
        iq_samples=samples,      # np.complex64 array, 60 seconds @ 20 kHz
        system_time=timestamp,   # Unix timestamp of buffer START
        rtp_timestamp=rtp_ts     # RTP timestamp of first sample
    )
    
    print(f"D_clock: {result.d_clock_ms:+.2f} ms")
    print(f"Mode: {result.solution.propagation_mode}")
    print(f"Uncertainty: {result.uncertainty_ms:.1f} ms")

================================================================================
OUTPUT: Phase2Result
================================================================================
The Phase2Result dataclass contains:
    - time_snap: TimeSnapResult (Step 1 output)
    - channel: ChannelCharacterization (Step 2 output)
    - solution: TransmissionTimeSolution (Step 3 output)
    - d_clock_ms: Final D_clock value
    - uncertainty_ms: Timing uncertainty in milliseconds
    - confidence: 0-1 confidence score

================================================================================
REVISION HISTORY
================================================================================
2025-12-07: Added comprehensive architectural documentation
2025-12-01: Integrated CHU FSK decoder for Canadian time signals
2025-11-20: Added test signal analysis for minutes 8/44
2025-11-01: Initial three-step architecture implementation
"""

import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple, Any, NamedTuple
from dataclasses import dataclass, field
import threading

logger = logging.getLogger(__name__)


# =============================================================================
# Constants for 32-bit Float Processing
# =============================================================================

# Phase 1 stores data as np.complex64 (32-bit float I + 32-bit float Q)
# This provides 144 dB dynamic range vs 96 dB for 16-bit int
EXPECTED_DTYPE = np.complex64
SAMPLE_RATE_FULL = 20000      # Phase 1 archive sample rate

# Decimation for tone detection
# Note: Decimation was considered but removed - tone detection now uses full rate
# for maximum timing accuracy. The matched filter templates are generated at
# the full sample rate (20 kHz).

# Normalization threshold for 32-bit float data
# Since AGC is disabled (F32 has sufficient dynamic range), we apply
# a fixed normalization to ensure consistent processing
MAX_EXPECTED_AMPLITUDE = 1.0  # Normalized float range [-1, 1]
AMPLITUDE_WARNING_THRESHOLD = 10.0  # Flag if amplitude exceeds this


@dataclass
class TimeSnapResult:
    """
    Result of Step 1: Fundamental Tone Detection & Time Snap Correction.
    
    This establishes the initial synchronization point for all subsequent analysis.
    """
    # Time snap anchor
    timing_error_ms: float           # Offset from expected second boundary
    arrival_rtp: int                 # RTP timestamp of detected tone arrival
    arrival_system_time: float       # System time of arrival
    
    # Tone detection results
    wwv_detected: bool
    wwvh_detected: bool
    chu_detected: bool = False
    wwv_snr_db: Optional[float] = None
    wwvh_snr_db: Optional[float] = None
    chu_snr_db: Optional[float] = None
    wwv_timing_ms: Optional[float] = None
    wwvh_timing_ms: Optional[float] = None
    chu_timing_ms: Optional[float] = None
    
    # Quality metrics
    anchor_station: str = 'UNKNOWN'  # Station used for time snap ('WWV', 'WWVH', 'CHU')
    anchor_confidence: float = 0.0
    search_window_ms: float = 500.0  # Initial search window (narrowed in Step 2)
    
    # Provenance
    detection_method: str = 'matched_filter'


@dataclass
class ChannelCharacterization:
    """
    Result of Step 2: Ionospheric Channel Characterization.
    
    Contains BCD correlation, Doppler, and station identity results.
    """
    # BCD Correlation (Step 2A)
    bcd_wwv_amplitude: Optional[float] = None
    bcd_wwvh_amplitude: Optional[float] = None
    bcd_differential_delay_ms: Optional[float] = None
    bcd_correlation_quality: Optional[float] = None
    bcd_wwv_toa_ms: Optional[float] = None    # Absolute ToA from minute start
    bcd_wwvh_toa_ms: Optional[float] = None
    
    # Doppler and Coherence (Step 2B)
    doppler_wwv_hz: Optional[float] = None
    doppler_wwvh_hz: Optional[float] = None
    doppler_wwv_std_hz: Optional[float] = None
    doppler_wwvh_std_hz: Optional[float] = None
    max_coherent_window_sec: Optional[float] = None
    doppler_quality: Optional[float] = None
    phase_variance_rad: Optional[float] = None
    
    # Channel multipath metrics
    delay_spread_ms: Optional[float] = None
    coherence_time_sec: Optional[float] = None
    spreading_factor: Optional[float] = None  # L = τ_D × f_D
    
    # Station Identity (Step 2C)
    dominant_station: str = 'UNKNOWN'
    station_confidence: str = 'low'
    ground_truth_station: Optional[str] = None  # From 500/600 Hz exclusive minutes
    ground_truth_source: Optional[str] = None   # '500Hz', '600Hz', '440Hz'
    ground_truth_power_db: Optional[float] = None  # Power of detected ground truth tone
    
    # Harmonic power ratios (500/600 Hz detection)
    harmonic_ratio_500_1000: Optional[float] = None  # P_1000/P_500 in dB
    harmonic_ratio_600_1200: Optional[float] = None  # P_1200/P_600 in dB
    
    # BCD Intermodulation analysis (Vote 13)
    # 400 Hz = 500-100 (WWV signature), 700 Hz = 600+100 (WWVH signature)
    intermod_power_400_hz_db: Optional[float] = None  # WWV BCD sideband
    intermod_power_700_hz_db: Optional[float] = None  # WWVH BCD sideband
    intermod_ratio_400_700_db: Optional[float] = None  # WWV vs WWVH intermod signature
    intermod_dominant_station: Optional[str] = None    # From intermod analysis
    intermod_confidence: float = 0.0
    
    # Test signal analysis (minutes 8 and 44 only)
    test_signal_detected: bool = False
    test_signal_fss_db: Optional[float] = None  # Frequency Selectivity Score (D-layer indicator)
    test_signal_delay_spread_ms: Optional[float] = None  # Multipath from chirp analysis
    test_signal_toa_offset_ms: Optional[float] = None  # High-precision ToA
    test_signal_coherence_time_sec: Optional[float] = None  # Channel stability
    
    # CHU FSK analysis (seconds 31-39 of each minute)
    chu_fsk_detected: bool = False
    chu_fsk_frames_decoded: int = 0  # Number of successfully decoded frames (max 9)
    chu_fsk_timing_offset_ms: Optional[float] = None  # Offset from 500ms boundary
    chu_fsk_dut1_seconds: Optional[float] = None  # UT1-UTC correction
    chu_fsk_tai_utc: Optional[int] = None  # TAI-UTC (leap seconds)
    chu_fsk_decode_confidence: float = 0.0  # Frame decode success rate
    chu_fsk_time_verified: bool = False  # Decoded time matches expected
    
    # Narrowed search window for Step 3
    refined_search_window_ms: float = 50.0  # Tightened from 500ms to 50ms
    
    # Carrier SNR for uncertainty estimation
    snr_db: Optional[float] = None  # Carrier signal-to-noise ratio
    
    # Validation
    cross_validation_agreements: List[str] = field(default_factory=list)
    cross_validation_disagreements: List[str] = field(default_factory=list)


@dataclass
class TransmissionTimeSolution:
    """
    Result of Step 3: Transmission Time Solution.
    
    The final D_clock output representing the clock offset.
    """
    # The Holy Grail: D_clock
    d_clock_ms: float               # D_clock = T_system - T_UTC
    
    # UTC recovery
    t_emission_ms: float            # Back-calculated emission time offset
    t_arrival_ms: float             # Measured arrival time
    t_propagation_ms: float         # Calculated propagation delay
    
    # Propagation mode identification
    propagation_mode: str           # '1F', '2F', 'GW', etc.
    n_hops: int                     # Number of ionospheric hops
    layer_height_km: float          # Estimated ionospheric layer height
    
    # Station used for solution
    station: str                    # 'WWV', 'WWVH', 'CHU'
    frequency_mhz: float
    
    # Confidence metrics
    confidence: float               # 0-1 overall confidence
    uncertainty_ms: float           # Estimated timing uncertainty
    utc_verified: bool = False      # True if |emission_offset| < 2ms
    
    # Dual-station cross-validation
    dual_station_agreement_ms: Optional[float] = None  # |T_wwv - T_wwvh|
    dual_station_verified: bool = False
    
    # All propagation mode candidates with probabilities (for Mode Ridge visualization)
    mode_candidates: List[Dict] = field(default_factory=list)


@dataclass
class Phase2Result:
    """
    Complete Phase 2 analysis result for one minute of data.
    
    Combines all three steps into a single output structure.
    """
    # Timing reference
    minute_boundary_utc: float      # UTC minute boundary this measurement relates to
    system_time: float              # System time of first sample
    rtp_timestamp: int              # RTP timestamp of first sample
    
    # Step 1: Time Snap
    time_snap: TimeSnapResult
    
    # Step 2: Channel Characterization
    channel: ChannelCharacterization
    
    # Step 3: Transmission Time Solution
    solution: TransmissionTimeSolution
    
    # Final D_clock (propagated from solution)
    d_clock_ms: float
    utc_time: float                 # Calculated UTC = system_time - d_clock
    
    # Quality metrics (Issue 6.2 Fix: replaced arbitrary grades with uncertainty)
    uncertainty_ms: float = 10.0    # Estimated timing uncertainty in ms
    confidence: float = 0.0         # 0-1 confidence score
    
    # Deprecated: quality_grade removed per Issue 6.2
    # Old grades (A/B/C/D/X) had no statistical basis and are replaced by
    # uncertainty_ms which has physical meaning (expected error bounds).
    
    # Processing metadata
    processing_version: str = '2.1.0'  # Version bump for grade removal
    processed_at: Optional[float] = None


class Phase2TemporalEngine:
    """
    Phase 2 Temporal Analysis Engine.
    
    Implements the refined temporal analysis order:
    1. Fundamental Tone Detection → Time Snap Anchor
    2. Ionospheric Channel Characterization → Confidence Scoring
    3. Transmission Time Solution → D_clock
    """
    
    def __init__(
        self,
        raw_archive_dir: Path,
        output_dir: Path,
        channel_name: str,
        frequency_hz: float,
        receiver_grid: str,
        sample_rate: int = SAMPLE_RATE_FULL,
        precise_lat: Optional[float] = None,
        precise_lon: Optional[float] = None
    ):
        """
        Initialize the Phase 2 Temporal Engine.
        
        Args:
            raw_archive_dir: Directory containing Phase 1 raw archive
            output_dir: Output directory for Phase 2 products
            channel_name: Channel identifier (e.g., 'WWV_10MHz')
            frequency_hz: Center frequency in Hz
            receiver_grid: Receiver Maidenhead grid square (e.g., 'EM38ww')
            sample_rate: Input sample rate (default 20000 Hz)
            precise_lat: Optional precise latitude (improves timing by ~16μs)
            precise_lon: Optional precise longitude (improves timing by ~16μs)
        """
        self.raw_archive_dir = Path(raw_archive_dir)
        self.output_dir = Path(output_dir)
        self.channel_name = channel_name
        self.frequency_hz = frequency_hz
        self.frequency_mhz = frequency_hz / 1e6
        self.receiver_grid = receiver_grid
        self.sample_rate = sample_rate
        self.precise_lat = precise_lat
        self.precise_lon = precise_lon
        
        # Initialize sub-components (lazy import to avoid circular deps)
        self._init_components()
        
        # Processing state
        self._lock = threading.Lock()
        self.minutes_processed = 0
        self.last_result: Optional[Phase2Result] = None
        
        logger.info(f"Phase2TemporalEngine initialized for {channel_name}")
        logger.info(f"  Frequency: {self.frequency_mhz:.2f} MHz")
        logger.info(f"  Receiver: {receiver_grid}")
        logger.info(f"  Sample rate: {sample_rate} Hz")
    
    def _init_components(self):
        """Initialize analysis sub-components."""
        try:
            # Step 1: Tone Detector - use FULL rate for accurate timing
            # Decimation causes timing errors due to spectral interactions
            from .tone_detector import MultiStationToneDetector
            self.tone_detector = MultiStationToneDetector(
                channel_name=self.channel_name,
                sample_rate=self.sample_rate  # Full rate (20 kHz) for accuracy
            )
            
            # Step 2: WWV/WWVH Discriminator (includes BCD and Doppler)
            from .wwvh_discrimination import WWVHDiscriminator
            self.discriminator = WWVHDiscriminator(
                channel_name=self.channel_name,
                receiver_grid=self.receiver_grid,
                sample_rate=self.sample_rate
            )
            
            # Step 3: Transmission Time Solver
            from .transmission_time_solver import (
                TransmissionTimeSolver,
                create_solver_from_grid
            )
            self.solver = create_solver_from_grid(
                self.receiver_grid,
                self.sample_rate,
                precise_lat=self.precise_lat,
                precise_lon=self.precise_lon
            )
            
            logger.info("✅ Phase 2 components initialized")
            
        except ImportError as e:
            logger.error(f"Failed to initialize Phase 2 components: {e}")
            raise
    
    def _validate_input(self, iq_samples: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Validate and normalize 32-bit float IQ input data.
        
        Ensures input is np.complex64 as expected from Phase 1 archive.
        Applies fixed normalization to prevent numerical issues while
        preserving linearity.
        
        Args:
            iq_samples: Input IQ samples from Phase 1 archive
            
        Returns:
            Tuple of (normalized_samples, validation_metrics)
        """
        metrics = {
            'input_dtype': str(iq_samples.dtype),
            'input_shape': iq_samples.shape,
            'max_amplitude': 0.0,
            'mean_amplitude': 0.0,
            'normalization_applied': False,
            'amplitude_warning': False
        }
        
        # Check dtype - must be complex64 (32-bit float IQ)
        if iq_samples.dtype != EXPECTED_DTYPE:
            logger.warning(
                f"Input dtype {iq_samples.dtype} differs from expected {EXPECTED_DTYPE}. "
                f"Converting to {EXPECTED_DTYPE}."
            )
            iq_samples = iq_samples.astype(EXPECTED_DTYPE)
        
        # Calculate amplitude statistics
        amplitudes = np.abs(iq_samples)
        max_amp = float(np.max(amplitudes))
        mean_amp = float(np.mean(amplitudes))
        
        metrics['max_amplitude'] = max_amp
        metrics['mean_amplitude'] = mean_amp
        
        # Check for amplitude warnings
        if max_amp > AMPLITUDE_WARNING_THRESHOLD:
            logger.warning(
                f"High amplitude detected: max={max_amp:.2f} > threshold={AMPLITUDE_WARNING_THRESHOLD}. "
                f"This may indicate decode errors or unusual signal conditions."
            )
            metrics['amplitude_warning'] = True
        
        # Apply fixed normalization if needed
        # Since Phase 1 uses F32 without AGC, the amplitude range varies
        # Normalize to [-1, 1] range for consistent processing
        if max_amp > MAX_EXPECTED_AMPLITUDE:
            normalization_factor = max_amp
            iq_samples = iq_samples / normalization_factor
            metrics['normalization_applied'] = True
            metrics['normalization_factor'] = normalization_factor
            logger.debug(f"Applied normalization: factor={normalization_factor:.4f}")
        
        return iq_samples, metrics
    
    def _step1_tone_detection(
        self,
        iq_samples: np.ndarray,
        system_time: float,
        rtp_timestamp: int
    ) -> TimeSnapResult:
        """
        Step 1: Fundamental Tone Detection & Time Snap Correction.
        
        Establishes the initial Time Snap Reference for the RTP stream by
        detecting 800ms timing tones (1000 Hz WWV, 1200 Hz WWVH).
        
        Args:
            iq_samples: Full-rate (20 kHz) complex64 IQ samples
            system_time: System time of first sample
            rtp_timestamp: RTP timestamp of first sample
            
        Returns:
            TimeSnapResult with initial timing anchor
        """
        # Calculate buffer mid-point timestamp
        # The tone detector expects timestamp at MIDDLE of buffer for
        # correct minute boundary calculation
        buffer_duration = len(iq_samples) / self.sample_rate
        buffer_mid_time = system_time + buffer_duration / 2
        
        # Run matched filter tone detection at FULL RATE (20 kHz)
        # The tone detector is initialized with self.sample_rate (full rate)
        # for maximum timing accuracy and sub-sample interpolation precision.
        # Full-rate detection provides more accurate timing than decimated.
        # Future optimization: implement proper anti-alias decimation.
        detections = self.tone_detector.process_samples(
            timestamp=buffer_mid_time,
            samples=iq_samples,  # Full rate for accuracy
            rtp_timestamp=rtp_timestamp,
            original_sample_rate=self.sample_rate,
            buffer_rtp_start=rtp_timestamp
        )
        
        # Extract detection results
        wwv_det = None
        wwvh_det = None
        chu_det = None
        
        if detections:
            from .interfaces.data_models import StationType
            for det in detections:
                if det.station == StationType.WWV:
                    wwv_det = det
                elif det.station == StationType.WWVH:
                    wwvh_det = det
                elif det.station == StationType.CHU:
                    chu_det = det
        
        # Determine anchor station (highest confidence detection)
        # Priority: WWV > CHU > WWVH (CHU uses same 1000 Hz as WWV)
        anchor_station = 'UNKNOWN'
        anchor_confidence = 0.0
        timing_error_ms = 0.0
        arrival_rtp = rtp_timestamp
        
        if wwv_det and wwvh_det:
            # Both WWV and WWVH detected - use higher confidence
            if wwv_det.confidence >= wwvh_det.confidence:
                anchor_station = 'WWV'
                anchor_confidence = wwv_det.confidence
                timing_error_ms = wwv_det.timing_error_ms or 0.0
            else:
                anchor_station = 'WWVH'
                anchor_confidence = wwvh_det.confidence
                timing_error_ms = wwvh_det.timing_error_ms or 0.0
        elif wwv_det:
            anchor_station = 'WWV'
            anchor_confidence = wwv_det.confidence
            timing_error_ms = wwv_det.timing_error_ms or 0.0
        elif chu_det:
            # CHU detected (500ms @ 1000 Hz) - valid time reference
            anchor_station = 'CHU'
            anchor_confidence = chu_det.confidence
            timing_error_ms = chu_det.timing_error_ms or 0.0
        elif wwvh_det:
            anchor_station = 'WWVH'
            anchor_confidence = wwvh_det.confidence
            timing_error_ms = wwvh_det.timing_error_ms or 0.0
        
        # Calculate arrival RTP from timing error
        # Use round() not int() for proper rounding (at 20kHz, 1 sample = 0.05ms)
        timing_offset_samples = round(timing_error_ms * self.sample_rate / 1000)
        arrival_rtp = rtp_timestamp + timing_offset_samples
        
        result = TimeSnapResult(
            timing_error_ms=timing_error_ms,
            arrival_rtp=arrival_rtp,
            arrival_system_time=system_time + (timing_error_ms / 1000.0),
            wwv_detected=wwv_det is not None,
            wwvh_detected=wwvh_det is not None,
            chu_detected=chu_det is not None,
            wwv_snr_db=wwv_det.snr_db if wwv_det else None,
            wwvh_snr_db=wwvh_det.snr_db if wwvh_det else None,
            chu_snr_db=chu_det.snr_db if chu_det else None,
            wwv_timing_ms=wwv_det.timing_error_ms if wwv_det else None,
            wwvh_timing_ms=wwvh_det.timing_error_ms if wwvh_det else None,
            chu_timing_ms=chu_det.timing_error_ms if chu_det else None,
            anchor_station=anchor_station,
            anchor_confidence=anchor_confidence,
            search_window_ms=500.0  # Initial wide window, narrowed in Step 2
        )
        
        logger.debug(
            f"Step 1 Time Snap: anchor={anchor_station}, "
            f"timing_error={timing_error_ms:+.2f}ms, confidence={anchor_confidence:.2f}"
        )
        
        return result
    
    def _step2_channel_characterization(
        self,
        iq_samples: np.ndarray,
        time_snap: TimeSnapResult,
        system_time: float,
        minute_number: int
    ) -> ChannelCharacterization:
        """
        Step 2: Ionospheric Channel Characterization.
        
        Uses the time snap from Step 1 to synchronize BCD correlation and
        Doppler estimation for high-sensitivity channel analysis.
        
        Sub-steps:
        A. BCD Correlation & Dual-Peak Delay
        B. Doppler and Coherence Estimation
        C. Station Identity & Ground Truth
        
        Args:
            iq_samples: Full-rate (20 kHz) complex64 IQ samples
            time_snap: Result from Step 1
            system_time: System time of first sample
            minute_number: Minute of hour (0-59)
            
        Returns:
            ChannelCharacterization with channel metrics
        """
        result = ChannelCharacterization()
        agreements = []
        disagreements = []
        
        # === Step 2A: BCD Correlation & Dual-Peak Delay ===
        # The time snap from Step 1 provides the expected minute boundary,
        # allowing accurate template synchronization
        try:
            bcd_result = self.discriminator.detect_bcd_discrimination(
                iq_samples=iq_samples,
                sample_rate=self.sample_rate,
                minute_timestamp=system_time,
                frequency_mhz=self.frequency_mhz
            )
            
            if bcd_result and bcd_result[0] is not None:
                wwv_amp, wwvh_amp, delay_ms, quality, windows = bcd_result
                result.bcd_wwv_amplitude = wwv_amp
                result.bcd_wwvh_amplitude = wwvh_amp
                result.bcd_differential_delay_ms = delay_ms
                result.bcd_correlation_quality = quality
                
                # Extract ToA and delay spread from windows if available
                if windows and len(windows) > 0:
                    # Use first high-quality window
                    for w in windows:
                        if w.get('wwv_toa_ms') is not None:
                            result.bcd_wwv_toa_ms = w['wwv_toa_ms']
                            result.bcd_wwvh_toa_ms = w['wwvh_toa_ms']
                            
                            # Extract delay spread from BCD correlation peak widths
                            # Use the delay spread of the dominant station
                            wwv_spread = w.get('wwv_delay_spread_ms')
                            wwvh_spread = w.get('wwvh_delay_spread_ms')
                            if wwv_spread is not None and wwvh_spread is not None:
                                # Use average of both stations' delay spreads
                                result.delay_spread_ms = (wwv_spread + wwvh_spread) / 2.0
                            elif wwv_spread is not None:
                                result.delay_spread_ms = wwv_spread
                            elif wwvh_spread is not None:
                                result.delay_spread_ms = wwvh_spread
                            break
                
                # Log with None-safe formatting
                logger.debug(
                    f"Step 2A BCD: WWV_amp={wwv_amp if wwv_amp is not None else 'None'}, "
                    f"WWVH_amp={wwvh_amp if wwvh_amp is not None else 'None'}, "
                    f"delay={delay_ms if delay_ms is not None else 'None'}ms, "
                    f"quality={quality if quality is not None else 'None'}"
                )
        except Exception as e:
            logger.warning(f"Step 2A BCD correlation failed: {e}")
        
        # === Step 2B: Doppler and Coherence Estimation ===
        # Measure ionospheric stability from per-tick phase tracking
        try:
            doppler_info = self.discriminator.estimate_doppler_shift_from_ticks(
                iq_samples=iq_samples,
                sample_rate=self.sample_rate
            )
            
            if doppler_info:
                result.doppler_wwv_hz = doppler_info.get('wwv_doppler_hz')
                result.doppler_wwvh_hz = doppler_info.get('wwvh_doppler_hz')
                result.doppler_wwv_std_hz = doppler_info.get('wwv_doppler_std_hz')
                result.doppler_wwvh_std_hz = doppler_info.get('wwvh_doppler_std_hz')
                result.max_coherent_window_sec = doppler_info.get('max_coherent_window_sec')
                result.doppler_quality = doppler_info.get('doppler_quality')
                result.phase_variance_rad = doppler_info.get('phase_variance_rad')
                
                # Estimate coherence time from Doppler standard deviation
                max_std = max(
                    result.doppler_wwv_std_hz or 0.0,
                    result.doppler_wwvh_std_hz or 0.0
                )
                if max_std > 0.001:
                    # τ_c ≈ 1 / (π × f_D)
                    result.coherence_time_sec = 1.0 / (np.pi * max_std)
                else:
                    result.coherence_time_sec = 60.0  # Stable channel
                
                logger.debug(
                    f"Step 2B Doppler: WWV={result.doppler_wwv_hz:+.4f}Hz, "
                    f"T_max={result.max_coherent_window_sec:.1f}s"
                )
        except Exception as e:
            logger.warning(f"Step 2B Doppler estimation failed: {e}")
        
        # === Step 2C: Station Identity & Ground Truth ===
        # Check for exclusive broadcast minutes (500/600 Hz tones)
        try:
            # Function returns 6 values: (detected, power_db, freq, station, harmonic_500_1000, harmonic_600_1200)
            gt_result = self.discriminator.detect_500_600hz_tone(
                iq_samples=iq_samples,
                sample_rate=self.sample_rate,
                minute_number=minute_number
            )
            gt_detected, gt_power, gt_freq, gt_station = gt_result[:4]
            harmonic_500_1000, harmonic_600_1200 = gt_result[4], gt_result[5]
            
            # Always store harmonic ratios when computed (useful for analysis)
            if harmonic_500_1000 is not None:
                result.harmonic_ratio_500_1000 = harmonic_500_1000
            if harmonic_600_1200 is not None:
                result.harmonic_ratio_600_1200 = harmonic_600_1200
            
            if gt_detected and gt_station:
                result.ground_truth_station = gt_station
                result.ground_truth_source = f'{gt_freq}Hz'
                result.ground_truth_power_db = gt_power
                agreements.append(f'ground_truth_{gt_station}_{gt_freq}Hz')
                logger.info(
                    f"Step 2C Ground Truth: {gt_station} confirmed via {gt_freq} Hz "
                    f"(power={gt_power:.1f}dB)"
                )
        except Exception as e:
            logger.debug(f"Step 2C ground truth detection: {e}")
        
        # Check 440 Hz tone for minutes 1 and 2
        if minute_number in [1, 2]:
            try:
                detected_440, power_440 = self.discriminator.detect_440hz_tone(
                    iq_samples=iq_samples,
                    sample_rate=self.sample_rate,
                    minute_number=minute_number
                )
                
                if detected_440:
                    if minute_number == 1:
                        result.ground_truth_station = 'WWVH'
                        result.ground_truth_source = '440Hz_min1'
                    else:  # minute 2
                        result.ground_truth_station = 'WWV'
                        result.ground_truth_source = '440Hz_min2'
                    result.ground_truth_power_db = power_440
                    agreements.append(f'440Hz_minute{minute_number}')
                    logger.info(
                        f"Step 2C Ground Truth: {result.ground_truth_station} confirmed via 440Hz "
                        f"(power={power_440:.1f}dB)"
                    )
            except Exception as e:
                logger.debug(f"440 Hz detection: {e}")
        
        # Detect test signal for minutes 8 and 44 (channel sounding)
        # This provides FSS, delay spread, and high-precision ToA for timing improvement
        if minute_number in [8, 44]:
            try:
                test_result = self.discriminator.test_signal_detector.detect(
                    iq_samples=iq_samples,
                    minute_number=minute_number,
                    sample_rate=self.sample_rate
                )
                
                if test_result.detected:
                    result.test_signal_detected = True
                    result.test_signal_fss_db = test_result.frequency_selectivity_db
                    result.test_signal_delay_spread_ms = test_result.delay_spread_ms
                    result.test_signal_toa_offset_ms = test_result.toa_offset_ms
                    result.test_signal_coherence_time_sec = test_result.coherence_time_sec
                    
                    # Use test signal delay spread if better than BCD estimate
                    if test_result.delay_spread_ms is not None:
                        if result.delay_spread_ms is None or test_result.delay_spread_ms < result.delay_spread_ms:
                            result.delay_spread_ms = test_result.delay_spread_ms
                    
                    # Use test signal coherence time if available
                    if test_result.coherence_time_sec is not None:
                        result.coherence_time_sec = test_result.coherence_time_sec
                    
                    expected_station = 'WWV' if minute_number == 8 else 'WWVH'
                    agreements.append(f'test_signal_{expected_station}')
                    
                    logger.info(
                        f"Step 2C Test Signal: {expected_station} detected, "
                        f"FSS={test_result.frequency_selectivity_db:.1f}dB, "
                        f"delay_spread={test_result.delay_spread_ms:.2f}ms"
                        if test_result.delay_spread_ms else
                        f"Step 2C Test Signal: {expected_station} detected, "
                        f"FSS={test_result.frequency_selectivity_db}dB"
                    )
            except Exception as e:
                logger.debug(f"Test signal detection: {e}")
        
        # CHU FSK detection (all minutes for CHU channels)
        # CHU transmits FSK time code at seconds 31-39 with precise 500ms boundaries
        if 'CHU' in self.channel_name.upper():
            try:
                from .chu_fsk_decoder import CHUFSKDecoder
                
                if not hasattr(self, 'chu_fsk_decoder'):
                    self.chu_fsk_decoder = CHUFSKDecoder(
                        sample_rate=self.sample_rate,
                        channel_name=self.channel_name
                    )
                
                fsk_result = self.chu_fsk_decoder.decode_minute(
                    iq_samples=iq_samples,
                    minute_boundary_unix=system_time
                )
                
                if fsk_result.detected:
                    result.chu_fsk_detected = True
                    result.chu_fsk_frames_decoded = fsk_result.frames_decoded
                    result.chu_fsk_timing_offset_ms = fsk_result.timing_offset_ms
                    result.chu_fsk_dut1_seconds = fsk_result.dut1_seconds
                    result.chu_fsk_tai_utc = fsk_result.tai_utc
                    result.chu_fsk_decode_confidence = fsk_result.decode_confidence
                    
                    # Verify decoded time matches expected
                    expected_minute = minute_number
                    if fsk_result.decoded_minute == expected_minute:
                        result.chu_fsk_time_verified = True
                        agreements.append('chu_fsk_time_match')
                    else:
                        disagreements.append('chu_fsk_time_mismatch')
                    
                    logger.info(
                        f"Step 2C CHU FSK: {fsk_result.frames_decoded}/9 frames, "
                        f"timing={fsk_result.timing_offset_ms:.3f}ms, "
                        f"DUT1={fsk_result.dut1_seconds}s"
                        if fsk_result.dut1_seconds else
                        f"Step 2C CHU FSK: {fsk_result.frames_decoded}/9 frames, "
                        f"timing={fsk_result.timing_offset_ms:.3f}ms"
                    )
            except Exception as e:
                logger.debug(f"CHU FSK detection: {e}")
        
        # Determine dominant station from weighted voting
        # Use finalize_discrimination for complete voting
        try:
            # Create base discrimination result
            base_result = self.discriminator.compute_discrimination(
                detections=[],  # Detections handled via SNR below
                minute_timestamp=system_time
            )
            
            # Populate power_ratio_db from Step 1 tone detection SNRs
            # This enables Vote 3 (carrier power ratio) in finalize_discrimination
            if time_snap.wwv_snr_db is not None and time_snap.wwvh_snr_db is not None:
                base_result.power_ratio_db = time_snap.wwv_snr_db - time_snap.wwvh_snr_db
            elif time_snap.wwv_snr_db is not None:
                base_result.power_ratio_db = 10.0  # WWV detected only
                base_result.dominant_station = 'WWV'
            elif time_snap.wwvh_snr_db is not None:
                base_result.power_ratio_db = -10.0  # WWVH detected only
                base_result.dominant_station = 'WWVH'
            
            # Finalize with all evidence
            final_result = self.discriminator.finalize_discrimination(
                result=base_result,
                minute_number=minute_number,
                bcd_wwv_amp=result.bcd_wwv_amplitude,
                bcd_wwvh_amp=result.bcd_wwvh_amplitude,
                tone_440_wwv_detected=(minute_number == 2 and result.ground_truth_station == 'WWV'),
                tone_440_wwvh_detected=(minute_number == 1 and result.ground_truth_station == 'WWVH')
            )
            
            result.dominant_station = final_result.dominant_station or 'UNKNOWN'
            result.station_confidence = final_result.confidence
            
            # Collect validation results
            if final_result.inter_method_agreements:
                agreements.extend(final_result.inter_method_agreements)
            if final_result.inter_method_disagreements:
                disagreements.extend(final_result.inter_method_disagreements)
                
        except Exception as e:
            logger.warning(f"Station discrimination failed: {e}")
        
        # Calculate spreading factor L = τ_D × f_D
        if result.delay_spread_ms is not None and result.coherence_time_sec is not None:
            if result.coherence_time_sec > 0.01:
                f_D_est = 1.0 / (np.pi * result.coherence_time_sec)
                result.spreading_factor = (result.delay_spread_ms / 1000.0) * f_D_est
        
        # Narrow search window based on Step 2 confidence
        if result.station_confidence == 'high':
            result.refined_search_window_ms = 10.0  # Very tight
        elif result.station_confidence == 'medium':
            result.refined_search_window_ms = 25.0
        else:
            result.refined_search_window_ms = 50.0  # Conservative
        
        result.cross_validation_agreements = agreements
        result.cross_validation_disagreements = disagreements
        
        # Populate SNR for uncertainty estimation
        # Use the dominant station's SNR or max of detected SNRs
        if time_snap.wwv_snr_db is not None and time_snap.wwvh_snr_db is not None:
            if result.dominant_station == 'WWV':
                result.snr_db = time_snap.wwv_snr_db
            elif result.dominant_station == 'WWVH':
                result.snr_db = time_snap.wwvh_snr_db
            else:
                result.snr_db = max(time_snap.wwv_snr_db, time_snap.wwvh_snr_db)
        elif time_snap.wwv_snr_db is not None:
            result.snr_db = time_snap.wwv_snr_db
        elif time_snap.wwvh_snr_db is not None:
            result.snr_db = time_snap.wwvh_snr_db
        elif time_snap.chu_snr_db is not None:
            result.snr_db = time_snap.chu_snr_db
        
        return result
    
    def _station_from_channel_name(self) -> str:
        """
        Derive the transmitting station from the channel name.
        
        Channel names like "WWV 15 MHz", "WWVH 10 MHz", "CHU 7.85 MHz"
        tell us exactly which station we're receiving.
        
        Returns:
            'WWV', 'WWVH', 'CHU', or 'UNKNOWN'
        """
        if not self.channel_name:
            return 'UNKNOWN'
        
        name_upper = self.channel_name.upper()
        
        # Check for CHU first (to avoid matching "CHU" in other strings)
        if 'CHU' in name_upper:
            return 'CHU'
        # Check for WWVH before WWV (WWVH contains WWV)
        elif 'WWVH' in name_upper:
            return 'WWVH'
        elif 'WWV' in name_upper:
            return 'WWV'
        else:
            return 'UNKNOWN'
    
    def _is_shared_frequency(self) -> bool:
        """
        Check if this channel is on a shared WWV/WWVH frequency.
        
        Shared frequencies: 2.5, 5, 10, 15 MHz
        WWV-only: 20, 25 MHz
        CHU-only: 3.33, 7.85, 14.67 MHz
        
        Only shared frequencies need discrimination logic.
        """
        # Shared WWV/WWVH frequencies in MHz
        shared_freqs = {2.5, 5.0, 10.0, 15.0}
        
        # Check if this channel's frequency is shared
        if self.frequency_mhz in shared_freqs:
            return True
        
        # Also check channel name for explicit WWVH prefix on shared freqs
        # (e.g., "WWVH 10 MHz" is unambiguous even though 10 MHz is shared)
        name_upper = self.channel_name.upper() if self.channel_name else ''
        if 'WWVH' in name_upper:
            return False  # Explicitly WWVH, no discrimination needed
        
        return False
    
    def _step3_transmission_time_solution(
        self,
        time_snap: TimeSnapResult,
        channel: ChannelCharacterization,
        system_time: float,
        rtp_timestamp: int
    ) -> TransmissionTimeSolution:
        """
        Step 3: Transmission Time Solution.
        
        Back-calculates the true T_emission (UTC) by accurately modeling the
        propagation delay using all high-confidence measurements from Steps 1 and 2.
        
        Args:
            time_snap: Result from Step 1 (timing anchor)
            channel: Result from Step 2 (channel metrics)
            system_time: System time of first sample
            rtp_timestamp: RTP timestamp of first sample
            
        Returns:
            TransmissionTimeSolution with final D_clock
        """
        # Determine which station to use for solution
        # 
        # Shared frequencies (discrimination needed): 2.5, 5, 10, 15 MHz
        # WWV-only: 20, 25 MHz
        # WWVH-only: (none in typical configs)
        # CHU-only: 3.33, 7.85, 14.67 MHz
        #
        # Priority 0: Non-shared channels - station is unambiguous from channel name
        station = None
        channel_station = self._station_from_channel_name()
        is_shared_frequency = self._is_shared_frequency()
        
        if not is_shared_frequency:
            # CHU, WWV 20/25 MHz, etc. - no discrimination needed
            station = channel_station
            logger.debug(f"Station = {station} (non-shared frequency, no discrimination)")
        
        # For shared frequencies only: use discrimination
        if not station:
            # Priority 1: Ground truth (500/600 Hz exclusive minutes, 440 Hz)
            if channel.ground_truth_station:
                station = channel.ground_truth_station
                logger.debug(f"Station from ground truth: {station}")
            
            # Priority 2: High confidence discrimination (detected via voting)
            elif channel.station_confidence == 'high' and channel.dominant_station not in ['UNKNOWN', 'BALANCED', None]:
                station = channel.dominant_station
                logger.debug(f"Station from discrimination (high confidence): {station}")
            
            # Priority 3: Medium confidence discrimination or channel name fallback
            elif channel.dominant_station not in ['UNKNOWN', 'BALANCED', 'NONE', None, '']:
                station = channel.dominant_station
                logger.debug(f"Station from discrimination (medium confidence): {station}")
            
            else:
                station = channel_station
                logger.debug(f"Station from channel name fallback: {station}")
        
        # Final fallback
        if not station or station in ['BALANCED', 'UNKNOWN', 'NONE', '']:
            station = 'WWV'
            logger.debug(f"Station fallback to WWV")
        
        # Prepare channel metrics for solver
        delay_spread_ms = channel.delay_spread_ms or 0.5
        doppler_std_hz = channel.doppler_wwv_std_hz if station == 'WWV' else channel.doppler_wwvh_std_hz
        doppler_std_hz = doppler_std_hz or 0.1
        
        # FSS from test signal detection (minutes 8 and 44 only)
        # This provides D-layer attenuation indicator for mode disambiguation
        fss_db = channel.test_signal_fss_db  # Will be None for non-test-signal minutes
        
        if fss_db is not None:
            logger.info(f"Using FSS={fss_db:.1f}dB from test signal for mode disambiguation")
        
        # Calculate expected second boundary RTP (minute boundary for all stations)
        # All stations (WWV, WWVH, CHU) transmit at second 0:
        # - WWV: 1000 Hz, 0.8s tone
        # - WWVH: 1200 Hz, 0.8s tone
        # - CHU: 1000 Hz, 0.5s tone (1.0s at top of hour)
        #
        # The expected_second_rtp is where the tone WOULD arrive if clock were perfect.
        # Calculate from system_time (buffer start) to minute boundary:
        minute_boundary = (int(system_time) // 60) * 60
        samples_to_boundary = int((minute_boundary - system_time) * self.sample_rate)
        expected_second_rtp = rtp_timestamp + samples_to_boundary
        
        # Get arrival RTP from time snap
        arrival_rtp = time_snap.arrival_rtp
        
        try:
            solver_result = self.solver.solve(
                station=station,
                frequency_mhz=self.frequency_mhz,
                arrival_rtp=arrival_rtp,
                delay_spread_ms=delay_spread_ms,
                doppler_std_hz=doppler_std_hz,
                fss_db=fss_db,
                expected_second_rtp=expected_second_rtp
            )
            
            # Extract D_clock
            d_clock_ms = solver_result.utc_nist_offset_ms or solver_result.emission_offset_ms
            
            # Convert mode candidates to dict format for serialization
            mode_candidates = [
                {
                    'mode': c.mode.value,
                    'delay_ms': round(c.total_delay_ms, 2),
                    'probability': round(c.plausibility, 3),
                    'n_hops': c.n_hops,
                    'elevation_deg': round(c.elevation_angle_deg, 1)
                }
                for c in solver_result.candidates
            ]
            
            solution = TransmissionTimeSolution(
                d_clock_ms=d_clock_ms,
                t_emission_ms=solver_result.emission_offset_ms,
                t_arrival_ms=time_snap.timing_error_ms,
                t_propagation_ms=solver_result.propagation_delay_ms,
                propagation_mode=solver_result.mode.value,
                n_hops=solver_result.n_hops,
                layer_height_km=solver_result.layer_height_km,
                station=station,
                frequency_mhz=self.frequency_mhz,
                confidence=solver_result.confidence,
                uncertainty_ms=self._calculate_uncertainty(solver_result, channel),
                utc_verified=solver_result.utc_nist_verified,
                mode_candidates=mode_candidates
            )
            
            # Check for dual-station cross-validation
            if channel.bcd_wwv_toa_ms is not None and channel.bcd_wwvh_toa_ms is not None:
                # Both stations detected - can cross-validate
                # Back-calculate emission time from each
                # (This would require expected delays from geo_predictor)
                pass  # TODO: Implement dual-station cross-validation
            
            logger.info(
                f"Step 3 Solution: D_clock={d_clock_ms:+.2f}ms, station={station}, "
                f"mode={solver_result.mode.value}, confidence={solver_result.confidence:.2f}"
            )
            
            return solution
            
        except Exception as e:
            logger.error(f"Step 3 TransmissionTimeSolver failed: {e}")
            
            # Return fallback solution with low confidence
            return TransmissionTimeSolution(
                d_clock_ms=time_snap.timing_error_ms,  # Use timing error as fallback
                t_emission_ms=0.0,
                t_arrival_ms=time_snap.timing_error_ms,
                t_propagation_ms=0.0,
                propagation_mode='UNKNOWN',
                n_hops=0,
                layer_height_km=0.0,
                station=station,
                frequency_mhz=self.frequency_mhz,
                confidence=0.1,
                uncertainty_ms=100.0,
                utc_verified=False
            )
    
    def _calculate_uncertainty(
        self,
        solver_result,
        channel: ChannelCharacterization
    ) -> float:
        """Calculate timing uncertainty based on all available metrics."""
        base_uncertainty = 2.0  # Base uncertainty in ms
        
        # Confidence scaling
        confidence_factor = 1.0 / max(solver_result.confidence, 0.1)
        
        # Channel quality scaling
        if channel.spreading_factor is not None:
            if channel.spreading_factor > 1.0:
                channel_factor = 3.0  # Overspread channel
            elif channel.spreading_factor > 0.3:
                channel_factor = 1.5  # Moderately spread
            else:
                channel_factor = 0.8  # Clean channel
        else:
            channel_factor = 1.2  # Unknown, assume moderate
        
        # Doppler stability
        max_std = max(
            channel.doppler_wwv_std_hz or 0.0,
            channel.doppler_wwvh_std_hz or 0.0
        )
        doppler_factor = 1.0 + (max_std * 2.0)  # Higher Doppler = more uncertainty
        
        uncertainty = base_uncertainty * confidence_factor * channel_factor * doppler_factor
        
        # BCD delay spread improvement: low delay spread = clean channel
        if channel.delay_spread_ms is not None:
            if channel.delay_spread_ms < 1.0:
                # Very clean channel - 20% improvement
                uncertainty *= 0.8
                logger.debug(f"BCD delay spread {channel.delay_spread_ms:.1f}ms reduces uncertainty by 20%")
            elif channel.delay_spread_ms < 2.0:
                # Clean channel - 10% improvement
                uncertainty *= 0.9
                logger.debug(f"BCD delay spread {channel.delay_spread_ms:.1f}ms reduces uncertainty by 10%")
            elif channel.delay_spread_ms > 5.0:
                # High multipath - increase uncertainty
                uncertainty *= 1.3
                logger.debug(f"BCD delay spread {channel.delay_spread_ms:.1f}ms increases uncertainty by 30%")
        
        # BCD correlation quality improvement
        if channel.bcd_correlation_quality is not None and channel.bcd_correlation_quality > 0.8:
            uncertainty *= 0.9  # High BCD quality = 10% improvement
            logger.debug(f"High BCD quality {channel.bcd_correlation_quality:.2f} reduces uncertainty by 10%")
        
        # Test signal improvement: reduce uncertainty when test signal provides
        # high-quality channel characterization
        if channel.test_signal_detected:
            test_signal_factor = 0.8  # 20% improvement from test signal
            
            # Further reduction if coherence time is good (stable channel)
            if channel.test_signal_coherence_time_sec is not None:
                if channel.test_signal_coherence_time_sec > 5.0:
                    test_signal_factor = 0.6  # Very stable channel
                elif channel.test_signal_coherence_time_sec > 2.0:
                    test_signal_factor = 0.7  # Stable channel
            
            uncertainty *= test_signal_factor
            logger.debug(f"Test signal reduces uncertainty by {(1-test_signal_factor)*100:.0f}%")
        
        # Ground truth provides additional confidence
        if channel.ground_truth_station is not None:
            uncertainty *= 0.9  # 10% improvement from ground truth confirmation
        
        # CHU FSK provides precise 500ms timing reference and time verification
        if channel.chu_fsk_detected:
            # Base improvement from FSK decode
            chu_factor = 0.8  # 20% improvement from FSK decode
            
            # Additional improvement based on decode confidence
            if channel.chu_fsk_decode_confidence > 0.8:
                chu_factor = 0.6  # 40% improvement for high confidence
            elif channel.chu_fsk_decode_confidence > 0.5:
                chu_factor = 0.7  # 30% improvement for moderate confidence
            
            # Extra boost if decoded time matches expected
            if channel.chu_fsk_time_verified:
                chu_factor *= 0.9  # Additional 10% for time verification
            
            uncertainty *= chu_factor
            logger.debug(f"CHU FSK reduces uncertainty by {(1-chu_factor)*100:.0f}% "
                        f"(confidence={channel.chu_fsk_decode_confidence:.2f})")
        
        return min(uncertainty, 100.0)  # Cap at 100ms
    
    def _estimate_uncertainty(
        self,
        solution: TransmissionTimeSolution,
        channel: ChannelCharacterization
    ) -> Tuple[float, float]:
        """
        Estimate timing uncertainty based on all available metrics.
        
        Issue 6.2 Fix: Replaces arbitrary letter grades (A/B/C/D) with
        a physically meaningful uncertainty estimate in milliseconds.
        
        Returns:
            (uncertainty_ms, confidence): Tuple of uncertainty and confidence
        """
        # Base uncertainty from propagation model
        base_uncertainty = 5.0  # ms, typical ionospheric timing uncertainty
        
        # Factors that reduce uncertainty
        has_ground_truth = channel.ground_truth_station is not None
        has_test_signal = channel.test_signal_detected
        has_chu_fsk = channel.chu_fsk_detected and channel.chu_fsk_time_verified
        agreement_count = len(channel.cross_validation_agreements)
        disagreement_count = len(channel.cross_validation_disagreements)
        
        # Start with solution's confidence
        confidence = solution.confidence
        uncertainty = base_uncertainty
        
        # CHU FSK provides verified timing (lowest uncertainty)
        if has_chu_fsk:
            uncertainty = 0.1  # 100 μs - FSK provides precise timing
            confidence = max(confidence, 0.95)
        
        # Ground truth minutes (silent minutes) provide known station
        elif has_ground_truth:
            uncertainty = 1.0  # 1 ms - known station eliminates ambiguity
            confidence = max(confidence, 0.9)
        
        # Test signal provides high SNR measurement
        elif has_test_signal:
            uncertainty = 2.0  # 2 ms
            confidence = max(confidence, 0.8)
        
        # Multiple method agreement reduces uncertainty
        elif agreement_count >= 3:
            uncertainty = 2.0
            confidence = max(confidence, 0.75)
        
        elif agreement_count >= 2:
            uncertainty = 3.0
            confidence = max(confidence, 0.6)
        
        # Disagreements increase uncertainty
        if disagreement_count > 0:
            uncertainty *= (1 + 0.5 * disagreement_count)
            confidence *= 0.8
        
        # Low SNR increases uncertainty
        if channel.snr_db is not None and channel.snr_db < 10:
            uncertainty *= 2.0
            confidence *= 0.7
        
        # Multipath increases uncertainty
        if channel.delay_spread_ms is not None and channel.delay_spread_ms > 1.0:
            uncertainty += channel.delay_spread_ms
        
        # Low solution confidence increases uncertainty
        if solution.confidence < 0.5:
            uncertainty *= (2.0 - solution.confidence)
        
        # Cap values
        uncertainty = min(uncertainty, 50.0)  # Max 50 ms
        confidence = max(0.0, min(1.0, confidence))
        
        return uncertainty, confidence
    
    def process_minute(
        self,
        iq_samples: np.ndarray,
        system_time: float,
        rtp_timestamp: int
    ) -> Optional[Phase2Result]:
        """
        Process one minute of IQ data through the complete Phase 2 pipeline.
        
        This is the main entry point for Phase 2 analysis, implementing the
        refined temporal analysis order:
        
        1. Fundamental Tone Detection → Time Snap Anchor
        2. Ionospheric Channel Characterization → Confidence Scoring
        3. Transmission Time Solution → D_clock
        
        Args:
            iq_samples: Complex64 IQ samples (60 seconds at sample_rate)
            system_time: System time of first sample (Unix timestamp)
            rtp_timestamp: RTP timestamp of first sample
            
        Returns:
            Phase2Result containing all analysis outputs and final D_clock,
            or None if analysis fails completely
        """
        # Calculate minute boundary
        minute_boundary = (int(system_time) // 60) * 60
        minute_number = int((system_time // 60) % 60)
        
        # Validate and normalize input
        iq_samples, validation_metrics = self._validate_input(iq_samples)
        
        if validation_metrics.get('amplitude_warning'):
            logger.warning(f"Input amplitude warning - proceeding with caution")
        
        try:
            # === STEP 1: Fundamental Tone Detection ===
            time_snap = self._step1_tone_detection(
                iq_samples=iq_samples,
                system_time=system_time,
                rtp_timestamp=rtp_timestamp
            )
            
            # === STEP 2: Ionospheric Channel Characterization ===
            channel = self._step2_channel_characterization(
                iq_samples=iq_samples,
                time_snap=time_snap,
                system_time=system_time,
                minute_number=minute_number
            )
            
            # === STEP 3: Transmission Time Solution ===
            solution = self._step3_transmission_time_solution(
                time_snap=time_snap,
                channel=channel,
                system_time=system_time,
                rtp_timestamp=rtp_timestamp
            )
            
            # Calculate final UTC time
            utc_time = system_time - (solution.d_clock_ms / 1000.0)
            
            # Estimate uncertainty (Issue 6.2 fix: replaced arbitrary grades)
            uncertainty_ms, confidence = self._estimate_uncertainty(solution, channel)
            
            # Assemble complete result
            result = Phase2Result(
                minute_boundary_utc=minute_boundary,
                system_time=system_time,
                rtp_timestamp=rtp_timestamp,
                time_snap=time_snap,
                channel=channel,
                solution=solution,
                d_clock_ms=solution.d_clock_ms,
                utc_time=utc_time,
                uncertainty_ms=uncertainty_ms,
                confidence=confidence,
                processing_version='2.1.0',
                processed_at=datetime.now(tz=timezone.utc).timestamp()
            )
            
            # Update state
            with self._lock:
                self.minutes_processed += 1
                self.last_result = result
            
            logger.info(
                f"Phase 2 complete: D_clock={solution.d_clock_ms:+.2f}ms, "
                f"uncertainty={uncertainty_ms:.1f}ms, station={solution.station}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Phase 2 processing failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        with self._lock:
            return {
                'minutes_processed': self.minutes_processed,
                'channel_name': self.channel_name,
                'frequency_mhz': self.frequency_mhz,
                'receiver_grid': self.receiver_grid,
                'last_d_clock_ms': self.last_result.d_clock_ms if self.last_result else None,
                'last_uncertainty_ms': self.last_result.uncertainty_ms if self.last_result else None,
                'last_confidence': self.last_result.confidence if self.last_result else None
            }


# =============================================================================
# Factory Function
# =============================================================================

def create_phase2_engine(
    raw_archive_dir: Path,
    output_dir: Path,
    channel_name: str,
    frequency_hz: float,
    receiver_grid: str,
    sample_rate: int = SAMPLE_RATE_FULL
) -> Phase2TemporalEngine:
    """
    Create a Phase 2 Temporal Engine with standard configuration.
    
    Args:
        raw_archive_dir: Directory containing Phase 1 raw archive
        output_dir: Output directory for Phase 2 products
        channel_name: Channel identifier
        frequency_hz: Center frequency in Hz
        receiver_grid: Receiver Maidenhead grid square
        sample_rate: Input sample rate (default 20000 Hz)
        
    Returns:
        Configured Phase2TemporalEngine
    """
    return Phase2TemporalEngine(
        raw_archive_dir=raw_archive_dir,
        output_dir=output_dir,
        channel_name=channel_name,
        frequency_hz=frequency_hz,
        receiver_grid=receiver_grid,
        sample_rate=sample_rate
    )
