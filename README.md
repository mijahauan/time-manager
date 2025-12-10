# time-manager

**Precision HF Time Transfer Daemon**

A standalone timing service that extracts UTC(NIST) from WWV/WWVH/CHU standard time broadcasts. It abstracts the ionospheric channel, providing clean D_clock and Station_ID to any consuming application.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What's New in v0.2.0

- **Per-station propagation delays**: Proper UTC(NIST) back-calculation using correct transmitter coordinates for WWV, WWVH, and CHU
- **IRI2020 ionospheric model**: Dynamic F2-layer heights for accurate propagation delay estimation
- **Chrony SHM integration**: Discipline system clock to sub-millisecond accuracy
- **GPS-disciplined RTP timestamps**: Uses actual ka9q-radio timestamps for sub-ms precision

See [CHANGELOG.md](CHANGELOG.md) for details.

## Overview

time-manager is **infrastructure**, not a science application. Just as `radiod` abstracts the SDR hardware, time-manager abstracts the Ionospheric Channel. It provides:

1. **D_clock**: System clock offset from UTC(NIST)
2. **Station identification**: WWV vs WWVH on shared frequencies
3. **Propagation mode**: 1F, 2F, ground wave estimation
4. **Uncertainty quantification**: Confidence bounds on all measurements

### Architecture

```
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
```

## Features

### Multi-Broadcast Fusion
Combines 13 broadcasts from 9 frequencies to achieve sub-millisecond accuracy:
- **WWV**: 2.5, 5, 10, 15, 20, 25 MHz
- **WWVH**: 2.5, 5, 10, 15 MHz (shared with WWV)
- **CHU**: 3.33, 7.85, 14.67 MHz

### Station Discrimination
8 independent methods to distinguish WWV from WWVH on shared frequencies:
- BCD correlation
- 1000/1200 Hz tone power ratio
- 500/600 Hz ground truth (exclusive minutes)
- 440 Hz station ID (minutes 1 & 2)
- Test signal (minutes 8 & 44)
- Geographic ToA prediction
- Doppler stability
- Harmonic analysis

### Chrony Integration
Feed chronyd via SHM refclock to discipline the entire Linux system clock. Any application gets "GPS-quality" timestamps automatically.

## Installation

```bash
# Clone or copy the repository
cd /home/wsprdaemon/time-manager

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install
pip install -e .

# For Chrony SHM support:
pip install -e ".[chrony]"
```

## Configuration

Create `/etc/time-manager/config.toml`:

```toml
[general]
data_root = "/tmp/grape-test"
sample_rate = 20000
poll_interval = 10.0

[receiver]
grid_square = "EM38ww"
latitude = 38.918461
longitude = -92.127974

[output]
shm_path = "/dev/shm/grape_timing"
enable_chrony = false
chrony_unit = 0

[channels]
enabled = [
    "WWV 2.5 MHz", "WWV 5 MHz", "WWV 10 MHz", "WWV 15 MHz",
    "WWV 20 MHz", "WWV 25 MHz",
    "CHU 3.33 MHz", "CHU 7.85 MHz", "CHU 14.67 MHz"
]
```

## Usage

### Start the Daemon

```bash
# Using config file
time-manager --config /etc/time-manager/config.toml

# Test mode
time-manager --data-root /tmp/grape-test --debug

# With Chrony integration
time-manager --enable-chrony
```

### Monitor Output

```bash
# Watch the shared memory output
watch -n1 'cat /dev/shm/grape_timing | jq .'

# Check Chrony sources (if enabled)
chronyc sources -v
```

### Consume from Applications

```python
from time_manager.output.shm_writer import SHMReader

reader = SHMReader('/dev/shm/grape_timing')

# Get D_clock
d_clock = reader.get_d_clock()
if d_clock is not None:
    print(f"Clock offset: {d_clock:+.2f} ms")

# Get station for a channel
station = reader.get_channel_station("WWV 10 MHz")
print(f"Station: {station}")  # "WWV" or "WWVH"

# Check if locked
if reader.is_locked():
    print("Time-manager has achieved lock")
```

## Chrony Configuration

Add to `/etc/chrony/chrony.conf`:

```
# HF Time Transfer via time-manager
refclock SHM 0 refid TMGR poll 6 precision 1e-4 offset 0.0 delay 0.2
```

Parameters:
- `poll 6`: Check every 64 seconds (2^6)
- `precision 1e-4`: 0.1ms precision estimate
- `delay 0.2`: Account for ~200ms ionospheric variation

Then restart chronyd:

```bash
sudo systemctl restart chronyd
```

Verify with:

```bash
chronyc sources -v
# Look for 'HF' reference
```

## Output Format

The shared memory file contains JSON:

```json
{
  "version": "1.0.0",
  "timestamp": 1733872800.0,
  "d_clock_ms": -1.25,
  "d_clock_uncertainty_ms": 0.55,
  "clock_status": "LOCKED",
  "fusion": {
    "contributing_broadcasts": 9,
    "total_broadcasts": 13,
    "fused_d_clock_ms": -1.25,
    "fusion_uncertainty_ms": 0.55
  },
  "channels": {
    "WWV_10_MHz": {
      "station": "WWV",
      "confidence": "high",
      "propagation_mode": "1F2",
      "d_clock_raw_ms": -1.30,
      "snr_db": 25.0
    }
  }
}
```

## Related Projects

- **[grape-recorder](https://github.com/mijahauan/grape-recorder)**: Science data recorder that consumes timing from time-manager
- **[ka9q-radio](https://github.com/ka9q/ka9q-radio)**: SDR server providing RTP streams
- **chronyd**: System clock discipline

## Documentation

- [CHANGELOG.md](CHANGELOG.md) - Version history and release notes
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Technical architecture and design

## License

MIT License - See [LICENSE](LICENSE) file

## Author

Michael James Hauan (AC0G)

## Repository

https://github.com/mijahauan/time-manager
