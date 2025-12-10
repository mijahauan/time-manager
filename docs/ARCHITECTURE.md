# time-manager Architecture

## Overview

time-manager is a precision HF time transfer daemon that extracts UTC(NIST) 
from WWV/WWVH/CHU standard time broadcasts. It provides clean timing data 
to consumer applications via shared memory and optionally disciplines the 
system clock via Chrony.

## Design Philosophy

### "Set, Monitor, Intervene"

With a GPSDO-disciplined SDR (like ka9q-radio), the hardware clock is already 
a secondary time standard. Instead of constant re-anchoring (which introduces 
propagation jitter), we:

1. **Set**: Establish time reference once at startup
2. **Monitor**: Project time forward by counting RTP samples, verify with tones
3. **Intervene**: Only re-anchor if discontinuity detected or physics violated

### Infrastructure vs Science

time-manager is **infrastructure**, not a science application:
- Like `radiod` abstracts SDR hardware → time-manager abstracts the ionospheric channel
- Consumers receive clean D_clock values without worrying about propagation modes
- Science applications (grape-recorder) focus on what they do best: recording data

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              time-manager                                    │
│                                                                              │
│  ┌─────────────────┐                                                         │
│  │  ka9q-radio     │  RTP multicast streams (9 channels)                     │
│  │  (radiod)       │──────────────────────────────────────┐                  │
│  └─────────────────┘                                      │                  │
│                                                           ▼                  │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        LiveTimeEngine                                   │ │
│  │                                                                         │ │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                │ │
│  │  │ ChannelBuffer│   │ ChannelBuffer│   │ ChannelBuffer│  ... (9 ch)   │ │
│  │  │ WWV 10 MHz   │   │ WWV 15 MHz   │   │ CHU 7.85 MHz │                │ │
│  │  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                │ │
│  │         │                  │                  │                        │ │
│  │         ▼                  ▼                  ▼                        │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │              Phase2TemporalEngine (per channel)                  │ │ │
│  │  │                                                                  │ │ │
│  │  │  ┌─────────────┐  ┌─────────────────┐  ┌──────────────────────┐ │ │ │
│  │  │  │ ToneDetector│  │ StationDiscrim. │  │ TransmissionTime    │ │ │ │
│  │  │  │ (100Hz tones)│  │ (WWV vs WWVH)  │  │ Solver              │ │ │ │
│  │  │  └─────────────┘  └─────────────────┘  └──────────────────────┘ │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  │                                   │                                    │ │
│  │                                   ▼                                    │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │              Multi-Broadcast Fusion                              │ │ │
│  │  │                                                                  │ │ │
│  │  │  • Per-broadcast calibration (learned offsets)                   │ │ │
│  │  │  • Weighted average (SNR × confidence × mode_quality)            │ │ │
│  │  │  • Outlier rejection (3σ from median)                            │ │ │
│  │  │  • Quality grading (A/B/C/D/F)                                   │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  │                                   │                                    │ │
│  └───────────────────────────────────┼────────────────────────────────────┘ │
│                                      │                                       │
│              ┌───────────────────────┴───────────────────────┐              │
│              ▼                                               ▼              │
│  ┌─────────────────────┐                       ┌─────────────────────────┐ │
│  │  SHM Writer         │                       │  Chrony SHM Writer      │ │
│  │                     │                       │                         │ │
│  │  /dev/shm/          │                       │  System V IPC           │ │
│  │  grape_timing.json  │                       │  key 0x4e545030         │ │
│  └─────────────────────┘                       └─────────────────────────┘ │
│              │                                               │              │
└──────────────┼───────────────────────────────────────────────┼──────────────┘
               │                                               │
               ▼                                               ▼
       ┌───────────────┐                             ┌─────────────────┐
       │ grape-recorder │                             │    chronyd      │
       │ (science app) │                             │ (system clock)  │
       └───────────────┘                             └─────────────────┘
```

## Key Components

### LiveTimeEngine (`engine/live_time_engine.py`)

The main orchestrator:
- Receives RTP streams via multicast
- Maintains per-channel buffers
- Runs Phase2 processing on minute boundaries
- Performs multi-broadcast fusion
- Updates SHM and Chrony outputs

### Phase2TemporalEngine (`timing/phase2_temporal_engine.py`)

Per-channel processing:
- Detects 100Hz timing tones
- Discriminates WWV vs WWVH on shared frequencies
- Calculates raw timing offset from minute boundary
- Estimates SNR and confidence

### TransmissionTimeSolver (`timing/transmission_time_solver.py`)

Propagation delay calculation:
- Computes ionospheric path geometry (1F, 2F, 3F modes)
- Uses dynamic layer heights from IRI2020 model
- Station-specific delays using correct coordinates
- Mode disambiguation based on observed delay

### IonosphericModel (`timing/ionospheric_model.py`)

Dynamic ionosphere parameters:
- IRI2020/IRI2016 for F2-layer heights (hmF2)
- Parametric fallback model
- Location and time-dependent calculations
- Caching for performance

### ChronySHM (`output/chrony_shm.py`)

System clock discipline:
- 96-byte NTP SHM structure
- Atomic count/valid synchronization
- Nanosecond precision timestamps
- Proper 64-bit alignment

## D_clock Calculation

The fundamental calculation to recover UTC(NIST):

```
D_clock = (tone_arrival_time - minute_boundary) - propagation_delay
        = timing_ms - propagation_delay_ms

Where:
  timing_ms         = When tone detected (system clock) - minute boundary
  propagation_delay = Time for signal to travel from transmitter to receiver
  
If system clock == UTC(NIST):
  timing_ms == propagation_delay
  D_clock == 0
  
If system clock is 5ms fast:
  timing_ms == propagation_delay + 5ms
  D_clock == +5ms
```

### Per-Station Propagation

Each station has unique geometry:

| Station | Location | Distance* | 2F Delay* |
|---------|----------|-----------|-----------|
| WWV | Fort Collins, CO (40.68°N, 105.04°W) | ~1600 km | ~7 ms |
| WWVH | Kekaha, HI (21.99°N, 159.76°W) | ~4000 km | ~17 ms |
| CHU | Ottawa, ON (45.29°N, 75.75°W) | ~3700 km | ~15 ms |

*Example for receiver in San Diego area

### Ionospheric Layer Heights

The propagation delay depends on ionospheric reflection height:

```
Path length = 2 × n_hops × √(hop_distance² + layer_height²)
Delay = path_length / speed_of_light
```

Layer heights from IRI2020:
- **E-layer**: ~100 km (daytime only)
- **F1-layer**: ~180 km (daytime only)  
- **F2-layer**: ~250-400 km (varies with solar activity, season, time of day)

## Multi-Broadcast Fusion

### Calibration Learning

Each broadcast learns a calibration offset via exponential moving average:
```python
offset_new = α × measurement + (1-α) × offset_old
α = 0.1 (slow learning to average out ionospheric variation)
```

### Weight Calculation

```python
weight = grade_weight × mode_weight × snr_weight

grade_weight: A=1.0, B=0.8, C=0.6, D=0.4, F=0.1
mode_weight: 1F=1.0, 2F=0.9, 3F=0.7, GW=1.2
snr_weight: SNR / 20.0 (capped at 1.0)
```

### Outlier Rejection

Remove measurements > 3σ from median before fusion.

### Fused Result

```python
d_clock_fused = Σ(weight_i × d_clock_calibrated_i) / Σ(weight_i)
uncertainty = 1 / √(Σ(weight_i))
```

## Output Interfaces

### Shared Memory JSON

Path: `/dev/shm/grape_timing`

```json
{
  "version": "1.0.0",
  "timestamp": 1733872800.0,
  "d_clock_ms": -1.25,
  "d_clock_uncertainty_ms": 0.55,
  "clock_status": "LOCKED",
  "fusion": {
    "contributing_broadcasts": 9,
    "d_clock_raw_ms": -1.30,
    "d_clock_calibrated_ms": -1.25
  },
  "channels": {
    "WWV_10_MHz": {
      "station": "WWV",
      "propagation_mode": "2F",
      "propagation_delay_ms": 6.83,
      "d_clock_ms": -1.30,
      "snr_db": 25.0
    }
  }
}
```

### Chrony SHM

System V IPC shared memory segment (key 0x4e545030 for unit 0).

chronyd reads reference_time (UTC from WWV) and system_time (when received),
calculates offset, and disciplines the system clock.

## Dependencies

### Required
- Python 3.11+
- numpy
- scipy
- ka9q-radio (RTP streaming)

### Optional
- **iri2020**: Best ionospheric modeling (requires gfortran)
- **iri2016**: Fallback ionospheric model
- **sysv-ipc**: Chrony SHM integration

## Files

```
time-manager/
├── src/time_manager/
│   ├── __init__.py
│   ├── main.py                    # Entry point
│   ├── engine/
│   │   └── live_time_engine.py    # Main orchestrator
│   ├── timing/
│   │   ├── phase2_temporal_engine.py
│   │   ├── transmission_time_solver.py
│   │   ├── tone_detector.py
│   │   ├── ionospheric_model.py
│   │   ├── wwv_geographic_predictor.py
│   │   └── wwv_constants.py
│   ├── output/
│   │   ├── shm_writer.py
│   │   └── chrony_shm.py
│   └── interfaces/
│       └── timing_result.py
├── config/
│   └── dev-config.toml
├── docs/
│   └── ARCHITECTURE.md
├── CHANGELOG.md
├── README.md
├── setup.py
└── requirements.txt
```
