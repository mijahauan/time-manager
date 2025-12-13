"""
Clock Convergence Model - Kalman Filter for GPSDO-disciplined systems

Ported from grape-recorder's clock_convergence.py

================================================================================
PHILOSOPHY: "SET, MONITOR, INTERVENTION"
================================================================================
Traditional approach: Constantly recalculate D_clock each minute
    Problem: Propagation variations appear as "noise" in the clock estimate

Our approach: Once clock is characterized, variations ARE the science
    1. SET: Converge to locked D_clock estimate (first 30 minutes)
    2. MONITOR: Track residuals as ionospheric propagation data
    3. INTERVENTION: Re-acquire only if physics violated

KEY INSIGHT:
    With a GPSDO (10⁻⁹ stability), the clock doesn't drift measurably in hours.
    Minute-to-minute D_clock variations are therefore NOT clock error—they
    are IONOSPHERIC PROPAGATION EFFECTS that we want to measure!

================================================================================
KALMAN FILTER MODEL
================================================================================
State vector: [offset_ms, drift_rate_ms_per_min]

For GPSDO systems:
    - Process noise (drift): ~0.0001 ms/min (GPSDO has no drift)
    - Measurement noise: ~20 ms (ionospheric variations)

This allows the filter to:
    - Track slow drift while filtering out random noise
    - Detect anomalies via innovation monitoring
    - Lock once uncertainty converges below threshold
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


class KalmanClockTracker:
    """
    Kalman filter for tracking clock offset with drift.
    
    This properly handles non-stationary clock behavior while filtering
    out ionospheric measurement noise.
    
    State vector: [offset_ms, drift_rate_ms_per_min]
    """
    
    def __init__(
        self,
        initial_offset_ms: float = 0.0,
        initial_uncertainty_ms: float = 100.0,
        process_noise_offset_ms: float = 0.01,
        process_noise_drift_ms_per_min: float = 0.0001,
        measurement_noise_ms: float = 20.0
    ):
        """
        Initialize Kalman filter.
        
        Args:
            initial_offset_ms: Initial guess for clock offset
            initial_uncertainty_ms: Initial uncertainty (large = uninformed prior)
            process_noise_offset_ms: Process noise for offset (oscillator noise)
            process_noise_drift_ms_per_min: Process noise for drift rate (GPSDO: ~0)
            measurement_noise_ms: Measurement noise (ionospheric: ~20ms)
        """
        # State vector: [offset, drift_rate]
        self.x = np.array([initial_offset_ms, 0.0])
        
        # State covariance matrix (initially very uncertain)
        self.P = np.array([
            [initial_uncertainty_ms**2, 0.0],
            [0.0, (initial_uncertainty_ms / 10)**2]
        ])
        
        # Process noise parameters
        self.q_offset = process_noise_offset_ms**2
        self.q_drift = process_noise_drift_ms_per_min**2
        
        # Measurement noise
        self.R = measurement_noise_ms**2
        
        # Measurement matrix: we only observe offset, not drift
        self.H = np.array([[1.0, 0.0]])
        
        # Tracking
        self.count = 0
        self.last_timestamp: Optional[float] = None
        self.innovation_history: List[float] = []
    
    @property
    def offset_ms(self) -> float:
        """Current offset estimate."""
        return float(self.x[0])
    
    @property
    def drift_rate_ms_per_min(self) -> float:
        """Current drift rate estimate (ms/minute)."""
        return float(self.x[1])
    
    @property
    def uncertainty_ms(self) -> float:
        """Current offset uncertainty (1-sigma)."""
        return float(np.sqrt(self.P[0, 0]))
    
    def predict(self, dt_minutes: float = 1.0) -> None:
        """
        Prediction step: project state forward in time.
        
        Uses constant-velocity model:
            offset(t+dt) = offset(t) + drift * dt
            drift(t+dt) = drift(t)
        """
        # State transition matrix
        F = np.array([
            [1.0, dt_minutes],
            [0.0, 1.0]
        ])
        
        # Process noise covariance (scaled by dt)
        Q = np.array([
            [self.q_offset + self.q_drift * dt_minutes**2 / 3, self.q_drift * dt_minutes / 2],
            [self.q_drift * dt_minutes / 2, self.q_drift]
        ]) * dt_minutes
        
        # Predict state and covariance
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
    
    def update(
        self,
        measurement_ms: float,
        timestamp: Optional[float] = None,
        measurement_noise_ms: Optional[float] = None
    ) -> Tuple[float, float, bool]:
        """
        Update step: incorporate new measurement.
        
        Args:
            measurement_ms: Measured clock offset
            timestamp: Unix timestamp (for dt calculation)
            measurement_noise_ms: Override measurement noise for this update
            
        Returns:
            (innovation, normalized_innovation, is_outlier)
        """
        # Calculate time delta for prediction
        if timestamp is not None and self.last_timestamp is not None:
            dt_minutes = (timestamp - self.last_timestamp) / 60.0
            dt_minutes = max(0.1, min(60.0, dt_minutes))  # Clamp to reasonable range
        else:
            dt_minutes = 1.0
        
        # Initialize with first measurement (not 0) to avoid large initial innovation
        if self.count == 0:
            self.x[0] = measurement_ms
            logger.info(f"Kalman initialized with first measurement: {measurement_ms:+.2f}ms")
        
        if self.last_timestamp is not None:
            self.predict(dt_minutes)
        
        self.last_timestamp = timestamp
        self.count += 1
        
        # Measurement noise (allow per-measurement override)
        R = (measurement_noise_ms**2) if measurement_noise_ms else self.R
        
        # Innovation (measurement residual)
        z = np.array([measurement_ms])
        y = z - self.H @ self.x
        innovation = float(y[0])
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + R
        S_scalar = float(S[0, 0])
        
        # Normalized innovation for outlier detection
        normalized_innovation = abs(innovation) / np.sqrt(S_scalar) if S_scalar > 0 else 0.0
        
        # Track innovation history
        self.innovation_history.append(innovation)
        if len(self.innovation_history) > 60:
            self.innovation_history.pop(0)
        
        # Check for outlier (don't update if extreme outlier)
        is_outlier = normalized_innovation > 5.0  # 5-sigma outlier
        
        if not is_outlier:
            # Kalman gain
            K = self.P @ self.H.T / S_scalar
            
            # Update state
            self.x = self.x + K.flatten() * innovation
            
            # Update covariance (Joseph form for numerical stability)
            I_KH = np.eye(2) - K @ self.H
            self.P = I_KH @ self.P @ I_KH.T + R * (K @ K.T)
        
        return innovation, normalized_innovation, is_outlier
    
    def get_std_dev(self) -> float:
        """Get standard deviation of recent innovations."""
        if len(self.innovation_history) < 5:
            return float('inf')
        return float(np.std(self.innovation_history))
    
    def reset(self, initial_offset_ms: float = 0.0):
        """Reset filter state (for reacquisition)."""
        self.x = np.array([initial_offset_ms, 0.0])
        self.P = np.array([
            [100.0**2, 0.0],
            [0.0, 10.0**2]
        ])
        self.count = 0
        self.last_timestamp = None
        self.innovation_history = []


class ConvergenceState(Enum):
    """State of the clock convergence model."""
    ACQUIRING = "acquiring"      # Building initial estimate
    CONVERGING = "converging"    # Uncertainty shrinking
    LOCKED = "locked"            # High confidence, monitoring mode
    REACQUIRE = "reacquire"      # Rebuilding after anomaly


@dataclass
class ConvergenceResult:
    """Result from processing a measurement."""
    d_clock_ms: float           # Best estimate
    uncertainty_ms: float       # Current uncertainty
    residual_ms: float          # Deviation from estimate (propagation science!)
    state: ConvergenceState
    is_locked: bool
    is_anomaly: bool
    convergence_progress: float  # 0.0 to 1.0
    innovation_ms: float        # Kalman innovation
    normalized_innovation: float


class ClockConvergenceModel:
    """
    Clock convergence model using Kalman filter.
    
    Implements "Set, Monitor, Intervention" philosophy:
    - Converges to stable D_clock estimate
    - Once locked, treats variations as propagation data
    - Re-acquires only on persistent anomalies
    """
    
    def __init__(
        self,
        lock_uncertainty_ms: float = 2.0,
        min_samples_for_lock: int = 30,
        anomaly_sigma: float = 3.0,
        max_consecutive_anomalies: int = 5
    ):
        """
        Initialize convergence model.
        
        Args:
            lock_uncertainty_ms: Lock when uncertainty < this value
            min_samples_for_lock: Minimum samples before locking
            anomaly_sigma: Sigma threshold for anomaly detection
            max_consecutive_anomalies: Trigger reacquire after this many
        """
        self.lock_uncertainty_ms = lock_uncertainty_ms
        self.min_samples_for_lock = min_samples_for_lock
        self.anomaly_sigma = anomaly_sigma
        self.max_consecutive_anomalies = max_consecutive_anomalies
        
        # Kalman filter
        # FIX 4 (2025-12-11): Reduced measurement_noise_ms from 20.0 to 5.0
        # for faster convergence. With Fix 2 pre-filtering bad measurements,
        # we can trust the input more and converge faster.
        # grape-recorder used 1.0, but 5.0 is more conservative for safety.
        self.kalman = KalmanClockTracker(
            initial_offset_ms=0.0,
            initial_uncertainty_ms=100.0,
            process_noise_offset_ms=0.01,
            process_noise_drift_ms_per_min=0.0001,  # GPSDO: nearly zero
            measurement_noise_ms=5.0                # Reduced for faster convergence
        )
        
        # State
        self.state = ConvergenceState.ACQUIRING
        self.locked_offset_ms: Optional[float] = None
        self.consecutive_anomalies = 0
    
    def process_measurement(
        self,
        d_clock_ms: float,
        timestamp: float,
        measurement_noise_ms: Optional[float] = None
    ) -> ConvergenceResult:
        """
        Process a new D_clock measurement.
        
        Args:
            d_clock_ms: Measured D_clock value
            timestamp: Unix timestamp
            measurement_noise_ms: Optional override for measurement noise
            
        Returns:
            ConvergenceResult with current state and estimates
        """
        # Update Kalman filter (initialization happens inside update() on first call)
        innovation, norm_innov, is_outlier = self.kalman.update(
            d_clock_ms, timestamp, measurement_noise_ms
        )
        
        # Current estimates
        offset_ms = self.kalman.offset_ms
        uncertainty_ms = self.kalman.uncertainty_ms
        count = self.kalman.count
        
        # Anomaly detection
        is_anomaly = norm_innov > self.anomaly_sigma
        
        if is_anomaly:
            self.consecutive_anomalies += 1
        else:
            self.consecutive_anomalies = 0
        
        # FIX 3 (2025-12-11): Reset on huge innovation during acquisition
        # If innovation is extremely large (>1000ms), the measurement is garbage
        # and would corrupt the filter. Reset immediately with the new measurement.
        if abs(innovation) > 1000.0 and self.state in (ConvergenceState.ACQUIRING, ConvergenceState.CONVERGING):
            logger.warning(f"Clock convergence: Huge innovation ({innovation:+.1f}ms) - resetting filter")
            self.kalman.reset(initial_offset_ms=d_clock_ms)
            self.consecutive_anomalies = 0
            # Re-run update to initialize properly
            innovation, norm_innov, is_outlier = self.kalman.update(
                d_clock_ms, timestamp, measurement_noise_ms
            )
            offset_ms = self.kalman.offset_ms
            uncertainty_ms = self.kalman.uncertainty_ms
            count = self.kalman.count
            is_anomaly = False
        
        # State machine transitions
        if self.state == ConvergenceState.ACQUIRING:
            if count >= 10:
                self.state = ConvergenceState.CONVERGING
                logger.info(f"Clock convergence: ACQUIRING -> CONVERGING (n={count})")
        
        elif self.state == ConvergenceState.CONVERGING:
            if uncertainty_ms < self.lock_uncertainty_ms and count >= self.min_samples_for_lock:
                self.state = ConvergenceState.LOCKED
                self.locked_offset_ms = offset_ms
                logger.info(f"Clock convergence: CONVERGING -> LOCKED "
                           f"(D_clock={offset_ms:+.2f}ms ±{uncertainty_ms:.2f}ms)")
        
        elif self.state == ConvergenceState.LOCKED:
            if self.consecutive_anomalies >= self.max_consecutive_anomalies:
                self.state = ConvergenceState.REACQUIRE
                logger.warning(f"Clock convergence: LOCKED -> REACQUIRE "
                              f"({self.consecutive_anomalies} consecutive anomalies)")
        
        elif self.state == ConvergenceState.REACQUIRE:
            # Reset and start over
            self.kalman.reset(initial_offset_ms=d_clock_ms)
            self.state = ConvergenceState.ACQUIRING
            self.consecutive_anomalies = 0
            logger.info("Clock convergence: REACQUIRE -> ACQUIRING (reset)")
        
        # Calculate convergence progress
        if count < 10:
            progress = count / 10 * 0.3  # 0-30% during acquiring
        elif self.state == ConvergenceState.CONVERGING:
            # Progress based on uncertainty reduction
            progress = 0.3 + 0.7 * max(0, 1 - uncertainty_ms / self.lock_uncertainty_ms)
        else:
            progress = 1.0
        
        # Residual (for propagation science when locked)
        residual_ms = d_clock_ms - offset_ms
        
        return ConvergenceResult(
            d_clock_ms=offset_ms,
            uncertainty_ms=uncertainty_ms,
            residual_ms=residual_ms,
            state=self.state,
            is_locked=(self.state == ConvergenceState.LOCKED),
            is_anomaly=is_anomaly,
            convergence_progress=min(1.0, progress),
            innovation_ms=innovation,
            normalized_innovation=norm_innov
        )
    
    @property
    def is_locked(self) -> bool:
        """True if model has locked onto stable estimate."""
        return self.state == ConvergenceState.LOCKED
    
    @property
    def d_clock_ms(self) -> float:
        """Current best D_clock estimate."""
        return self.kalman.offset_ms
    
    @property
    def uncertainty_ms(self) -> float:
        """Current uncertainty."""
        return self.kalman.uncertainty_ms
    
    def reset(self, initial_offset_ms: float = 0.0) -> None:
        """
        Reset model state (externally forced).
        
        Args:
            initial_offset_ms: New initial clock offset guess
        """
        self.kalman.reset(initial_offset_ms=initial_offset_ms)
        self.state = ConvergenceState.ACQUIRING
        self.consecutive_anomalies = 0
        self.locked_offset_ms = None
        logger.info(f"Clock convergence: Reset with initial offset {initial_offset_ms:+.2f}ms")
