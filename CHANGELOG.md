# Changelog

All notable changes to time-manager will be documented in this file.

## [0.3.0] - 2025-12-10

### Code Quality & Infrastructure Improvements

This release focuses on code correctness, test coverage, and operational infrastructure
following a comprehensive code review.

### Added

- **Unit Test Suite**: 30 comprehensive tests covering critical modules:
  - `test_chrony_shm.py`: Struct packing, size verification, field offsets
  - `test_transmission_time_solver.py`: Spherical geometry, ionospheric delays
  - `test_live_time_engine.py`: Buffer logic, state management, paths
  - `test_health_server.py`: HTTP endpoints, Prometheus metrics

- **Health Monitoring Server** (`output/health_server.py`):
  - `GET /health` - Basic health check (200 OK)
  - `GET /status` - JSON timing status with all metrics
  - `GET /metrics` - Prometheus-compatible metrics export
  - Integrated into main.py with `--health-port` argument (default: 8080)

- **Modern Python Packaging** (`pyproject.toml`):
  - Proper dependency management with optional extras
  - `pip install -e ".[dev]"` for development
  - `pip install -e ".[all]"` for all features
  - pytest configuration included

### Fixed

- **Chrony SHM Struct Size**: Corrected from 96 to **92 bytes** to match actual
  chronyd `struct shmTime` on 64-bit Linux with native alignment.
  ```python
  # Correct format: 92 bytes
  struct.pack('@iiqiqiiiiiii8i', ...)
  ```

- **Spherical Earth Geometry**: Fixed elevation angle calculation in
  `_calculate_hop_path()` for paths ≥500 km. Previous flat-Earth approximation
  caused 1-3% timing errors on long paths (WWV→Hawaii: ~4000 km).
  - Elevation angles now correctly decrease with distance
  - Uses law of cosines for slant range calculation
  - Uses law of sines for elevation angle derivation

- **Ionospheric Delay Factors**: Corrected to use proper 1/f² physics:
  ```python
  # Before (incorrect linear approximation):
  2.5 MHz: 1.5×
  
  # After (correct inverse-square):
  2.5 MHz: 16.0×  # (10/2.5)² = 16
  ```

- **Path References**: Updated all `grape-recorder` paths to `time-manager`:
  - State file: `/var/lib/time-manager/state/time_state.json`
  - Calibration: `/var/lib/time-manager/state/broadcast_calibration.json`

- **Ring Buffer RTP Tracking**: Fixed `full_start_rtp` assignment to occur
  before incrementing write position (was never being set).

- **Missing Import**: Added `import mmap` to `chrony_shm.py`.

- **Missing Function**: Added `channel_name_to_dir()` to `main.py`.

### Changed

- **Sample Rate Constant**: Renamed `SAMPLE_RATE_FULL` → `SAMPLE_RATE` (20000 Hz),
  removed deprecated `SAMPLE_RATE_LEGACY`.

### Technical Details

#### Spherical Earth Geometry

For ionospheric hop calculations on paths ≥500 km:

```
Given:
  R_e = 6371 km (Earth radius)
  R_layer = R_e + layer_height (e.g., 300 km for F2)
  θ = ground_distance / R_e (central angle)

Slant range (law of cosines):
  slant² = R_e² + R_layer² - 2·R_e·R_layer·cos(θ/2)

Elevation angle (law of sines):
  sin(angle_at_layer) = R_e·sin(θ/2) / slant
  elevation = π/2 - θ/2 - angle_at_layer
```

#### Test Coverage

```
tests/
├── conftest.py              # Shared fixtures
├── test_chrony_shm.py       # 6 tests
├── test_health_server.py    # 7 tests  
├── test_live_time_engine.py # 8 tests
└── test_transmission_time_solver.py  # 9 tests
```

Run with: `pytest tests/ -v`

---

## [0.2.0] - 2025-12-10

### Major Changes: UTC(NIST) Back-Calculation & Chrony Integration

This release focuses on achieving sub-millisecond precision through proper 
UTC(NIST) back-calculation and robust Chrony SHM integration.

### Added

- **Per-Station Propagation Delays**: Each broadcast (WWV, WWVH, CHU) now 
  calculates its own propagation delay using correct transmitter coordinates 
  and ionospheric geometry, instead of crude scaling factors.
  - WWV (Fort Collins, CO): ~1600 km path
  - WWVH (Kekaha, HI): ~4000 km path  
  - CHU (Ottawa, ON): ~3700 km path

- **IRI2020 Ionospheric Model**: Upgraded from IRI2016 to IRI2020 for more 
  accurate F2-layer height predictions. Falls back to IRI2016 or parametric 
  model if unavailable.

- **TransmissionTimeSolver.get_station_propagation_delay()**: New simplified 
  API for computing propagation delay for any station without full mode 
  disambiguation.

- **GPS-disciplined RTP Timestamps**: Now uses actual RTP timestamps from 
  ka9q-radio's StreamQuality (first_rtp_timestamp + batch_start_sample) 
  instead of sample counts, essential for sub-ms accuracy.

### Fixed

- **Chrony SHM Struct Layout**: Fixed 96-byte NTP SHM structure with proper 
  alignment and nanosecond fields for 64-bit Linux:
  ```
  @ii q i xxxx q i xxxx iiii II 8i  (96 bytes total)
  ```
  - Added receiveTimeStampNSec and clockTimeStampNSec fields
  - Added proper padding for 8-byte alignment
  - Atomic count increment for valid/count synchronization

- **WWVH Propagation Delay**: Previously used `WWV_delay * 2.5` approximation.
  Now calculates proper ionospheric path using WWVH coordinates (21.99°N, 
  159.76°W).

- **CHU Propagation Delay**: Previously used WWV's delay (wrong continent!).
  Now calculates proper path to Ottawa (45.29°N, 75.75°W).

- **Chrony Update Parameters**: Fixed _update_chrony() to pass reference_time 
  (UTC from WWV) and system_time correctly, with precision parameter.

### Technical Details

#### D_clock Calculation Flow

```
D_clock = tone_arrival_time - propagation_delay - minute_boundary

Where:
  tone_arrival_time : System time when 100Hz tone detected (from RTP timestamp)
  propagation_delay : Station-specific ionospheric path delay (ms)
  minute_boundary   : UTC second :00 when tone was transmitted at station

Result:
  D_clock = 0 means system clock perfectly aligned with UTC(NIST)
  D_clock > 0 means system clock is fast (ahead of UTC)
  D_clock < 0 means system clock is slow (behind UTC)
```

#### Propagation Delay Calculation

For each station, the solver:
1. Looks up transmitter coordinates from STATION_LOCATIONS
2. Calculates great-circle distance to receiver
3. Determines ionospheric layer heights (hmE, hmF2) from IRI2020 model
4. Computes hop geometry based on propagation mode (1F, 2F, 3F)
5. Calculates total path length and converts to delay

#### Chrony SHM Protocol

The 96-byte struct matches chronyd's expectations:
```c
struct shmTime {
    int mode;                    // 0=invalid, 1=valid
    int count;                   // Incremented atomically
    time_t clockTimeStampSec;    // System time (8 bytes on 64-bit)
    int clockTimeStampUSec;      // Microseconds
    int clockTimeStampNSec;      // Nanoseconds (new)
    // 4 bytes padding
    time_t receiveTimeStampSec;  // Reference time (8 bytes)
    int receiveTimeStampUSec;    // Microseconds
    int receiveTimeStampNSec;    // Nanoseconds (new)
    // 4 bytes padding
    int leap;                    // Leap second indicator
    int precision;               // log2(precision in seconds)
    int nsamples;                // Always 0
    int valid;                   // 0=invalid, 1=valid
    unsigned clockTimeStampFrac; // Sub-second fraction
    unsigned receiveTimeStampFrac;
    int pad[8];                  // Padding to 96 bytes
};
```

### Configuration

Recommended Chrony configuration for time-manager:
```
refclock SHM 0 refid TMGR poll 6 precision 1e-4 offset 0.0 delay 0.2
```

### Dependencies

- **iri2020**: Install from GitHub for best ionospheric modeling:
  ```bash
  pip install git+https://github.com/space-physics/iri2020.git
  ```
  Requires gfortran for compilation.

- **sysv_ipc**: Required for Chrony SHM integration:
  ```bash
  pip install sysv-ipc
  ```

## [0.1.0] - 2025-12-09

### Initial Release

- Multi-channel live streaming from ka9q-radio
- Per-channel Phase2 temporal processing
- Multi-broadcast fusion with calibration
- Shared memory output for client applications
- Basic Chrony SHM support
