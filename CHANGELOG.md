# Changelog

All notable changes to time-manager will be documented in this file.

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
