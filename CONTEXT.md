# time-manager AI Agent Context

This document provides context for AI agents working on the time-manager codebase.
Update this file as the project evolves.

---

## Project Overview

**time-manager** is a precision HF time transfer daemon that extracts UTC(NIST) from
WWV/WWVH/CHU radio broadcasts. It was carved out from `grape-recorder` to be a
standalone timing solution.

### Mission
Monitor time standard stations (WWV, WWVH, CHU) and their time pulses, calculate
propagation delays through the ionosphere, and determine system clock offset (D_clock)
relative to UTC(NIST) with sub-millisecond precision.

### Key Outputs
- **D_clock**: System clock offset from UTC(NIST) in milliseconds
- **Chrony SHM**: Feed timing to chronyd for system clock discipline
- **JSON Status**: Shared memory and HTTP endpoints for monitoring

---

## Current State (v0.3.0 - December 2025)

### What Works
- ✅ Live RTP streaming from ka9q-radio via ka9q-python
- ✅ Twin-Stream architecture (Fast Loop at :01, Slow Loop at :02)
- ✅ Multi-station tone detection (WWV, WWVH, CHU)
- ✅ Spherical Earth geometry for propagation delay calculation
- ✅ Chrony SHM integration (92-byte struct, correct alignment)
- ✅ Per-broadcast calibration with EMA learning
- ✅ Health monitoring HTTP server (Prometheus metrics)
- ✅ 30 unit tests passing

### What's Missing / TODO
- ❌ **Discrimination web page** - needs migration from grape-recorder
- ❌ **Time status web page** - needs migration from grape-recorder
- ❌ Web server infrastructure for serving pages
- ❌ WebSocket or SSE for real-time updates to web clients

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         time-manager                             │
├─────────────────────────────────────────────────────────────────┤
│  LiveTimeEngine (engine/live_time_engine.py)                    │
│  ├── RTP Subscription (ka9q-python RadiodStream)                │
│  ├── ChannelBuffer (ring + full minute buffers)                 │
│  ├── Fast Loop (:01) - Quick D_clock using previous state       │
│  └── Slow Loop (:02) - Full Phase2 analysis, state update       │
├─────────────────────────────────────────────────────────────────┤
│  Timing Pipeline (timing/)                                       │
│  ├── tone_detector.py - Matched filtering, onset detection      │
│  ├── wwvh_discrimination.py - 8-method weighted voting          │
│  ├── transmission_time_solver.py - Propagation delay calc       │
│  ├── phase2_temporal_engine.py - Full minute analysis           │
│  └── ionospheric_model.py - IRI2020/parametric layer heights    │
├─────────────────────────────────────────────────────────────────┤
│  Output Adapters (output/)                                       │
│  ├── chrony_shm.py - Chrony SHM refclock driver                 │
│  └── health_server.py - HTTP /health, /status, /metrics         │
├─────────────────────────────────────────────────────────────────┤
│  Interfaces (interfaces/)                                        │
│  └── timing_result.py - Data models for JSON serialization      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Files for Migration Work

### Source Files to Understand

| File | Purpose |
|------|---------|
| `src/time_manager/engine/live_time_engine.py` | Main engine, has `last_fusion`, `channel_states`, `d_clock_ms` |
| `src/time_manager/timing/wwvh_discrimination.py` | 8-method discrimination logic |
| `src/time_manager/interfaces/timing_result.py` | Data models: `TimingResult`, `ChannelTimingResult`, `DiscriminationInfo` |
| `src/time_manager/output/health_server.py` | HTTP server pattern to extend |

### grape-recorder Files to Migrate From

The discrimination and time pages live in grape-recorder. Key files:

```
/home/wsprdaemon/grape-recorder/
├── grape_recorder/
│   └── web/                    # Web server and pages
│       ├── server.py           # Flask/aiohttp server
│       ├── templates/          # HTML templates
│       │   ├── discrimination.html
│       │   └── time.html
│       └── static/             # JS, CSS assets
└── ...
```

### Data Available for Web Pages

From `LiveTimeEngine`:

```python
# Per-channel state (channel_states: Dict[str, ChannelState])
ChannelState:
    station: str              # "WWV", "WWVH", "CHU"
    propagation_mode: str     # "1F", "2F", "3F", etc.
    propagation_delay_ms: float
    snr_db: float
    confidence: float
    last_update_minute: int

# Fusion result (last_fusion: FusionResult)
FusionResult:
    d_clock_ms: float
    d_clock_raw_ms: float
    uncertainty_ms: float
    n_broadcasts: int
    quality_grade: str        # "A", "B", "C", "D"
    wwv_count: int
    wwvh_count: int
    chu_count: int

# Global timing
d_clock_ms: float
d_clock_uncertainty_ms: float
clock_drift_ppm: float
```

From `wwvh_discrimination.py`:

```python
# Discrimination methods and their weights
DISCRIMINATION_METHODS = [
    'tone_power_ratio',       # 100/500/600 Hz power analysis
    'differential_delay',     # Propagation time difference
    'tone_440hz',            # WWVH-only 440 Hz marker
    'ground_truth_500_600',  # Minute-specific tones
    'bcd_correlation',       # Time code matching
    'test_signal',           # Test signal detection
    'doppler_stability',     # Doppler shift analysis
    'harmonic_ratio'         # Harmonic power analysis
]

# Each method returns:
MethodResult:
    station: str             # "WWV" or "WWVH"
    confidence: float        # 0.0 to 1.0
    weight: float           # Method weight
    details: dict           # Method-specific data
```

---

## Migration Strategy for Web Pages

### Recommended Approach

1. **Extend health_server.py** or create new `web_server.py`:
   - Add routes for `/discrimination` and `/time` pages
   - Serve static HTML/JS/CSS
   - Add WebSocket or SSE endpoint for real-time updates

2. **Data Flow**:
   ```
   LiveTimeEngine
       ↓ (reference)
   WebServer.set_engine(engine)
       ↓ (polling or callback)
   JSON API endpoints
       ↓ (fetch/WebSocket)
   Browser JavaScript
   ```

3. **Page Requirements**:

   **Discrimination Page**:
   - Show all active channels (WWV 2.5/5/10/15/20/25, WWVH 2.5/5/10/15, CHU 3.33/7.85/14.67)
   - For shared frequencies (2.5, 5, 10, 15 MHz): show WWV vs WWVH discrimination
   - Display each discrimination method's vote and confidence
   - Show final weighted decision
   - Real-time updates (1-second refresh or WebSocket)

   **Time Page**:
   - Current D_clock with uncertainty
   - Quality grade (A/B/C/D)
   - Per-channel timing contributions
   - Historical D_clock graph (last N minutes)
   - Chrony sync status
   - Station propagation delays

### API Endpoints to Add

```
GET /api/timing          # Current timing state (JSON)
GET /api/discrimination  # Current discrimination state (JSON)
GET /api/channels        # Per-channel status (JSON)
WS  /ws/timing          # WebSocket for real-time updates
```

---

## Development Setup

```bash
cd /home/mjh/git/time-manager
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run in live mode (requires ka9q-radio)
python -m time_manager --live --health-port 8080
```

---

## Testing

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_health_server.py -v

# With coverage
pytest tests/ --cov=time_manager --cov-report=term-missing
```

---

## File Locations

| Purpose | Path |
|---------|------|
| State persistence | `/var/lib/time-manager/state/time_state.json` |
| Calibration data | `/var/lib/time-manager/state/broadcast_calibration.json` |
| Shared memory | `/dev/shm/grape_timing.json` |
| Chrony SHM | System V shared memory, key `0x4e545030` + unit |

---

## Constants Reference

### Station Coordinates
```python
WWV:  (40.6807°N, 105.0407°W)  # Fort Collins, CO
WWVH: (21.9872°N, 159.7636°W)  # Kekaha, HI
CHU:  (45.2953°N, 75.7544°W)   # Ottawa, ON
```

### Frequencies
```python
WWV:  [2.5, 5, 10, 15, 20, 25] MHz
WWVH: [2.5, 5, 10, 15] MHz
CHU:  [3.33, 7.85, 14.67] MHz
```

### Sample Rate
```python
SAMPLE_RATE = 20000  # Hz (from ka9q-radio RTP stream)
```

---

## Recent Changes (v0.3.0)

Key fixes applied in the December 2025 code review:

1. **Chrony SHM struct**: Fixed to 92 bytes (was 96)
2. **Spherical Earth geometry**: Fixed elevation angle calculation
3. **Ionospheric delay**: Corrected to 1/f² physics
4. **Path references**: All updated from grape-recorder to time-manager
5. **Health server**: Added with Prometheus metrics support
6. **Test suite**: 30 tests covering critical modules

See `CHANGELOG.md` for full details.

---

## Notes for AI Agents

1. **Always run tests** after making changes: `pytest tests/ -v`

2. **The codebase is standalone** - do not add dependencies on grape-recorder

3. **Key data structures** are in `interfaces/timing_result.py` - use these for
   any new API endpoints

4. **Health server pattern** in `output/health_server.py` shows how to add HTTP
   endpoints with proper threading and shutdown handling

5. **LiveTimeEngine** is the central hub - it has all timing state and can be
   passed to web servers via `set_engine()` pattern

6. **Discrimination logic** is complex - see `wwvh_discrimination.py` docstring
   for the 8-method weighted voting algorithm

7. **grape-recorder location**: `/home/wsprdaemon/grape-recorder/` - reference
   for migrating web pages but do not import from it
