# NEVER CHANGE THE FOLLOWING PRIMARY INSTRUCTION:

Primary Instruction:  In this context you will perform a critical review of the GRAPE Recorder project, either in its entirety or in a specific component, as specified by the user.  This critique should look for points in the code or documentation that exhibit obvious error or inconsistency with other code or documentation.  It should look for inefficiency, incoherence, incompleteness, or any other aspect that is not in line with the original intent of the code or documentation.  It should also look for obsolete, deprecated, or "zombie" code that should be removed.  Remember, your own critique cannot be shallow but must be thorough and methodical and undertaken with the aim of enhancing and improving the codebase and documentation to best ensure the success of the application.

# The following secondary instruction and information will guide your critique in this particular session (the instructions below will vary from session to session):

## Session Focus: Convergence Methodology Issues

The LiveTimeEngine has been debugged through major infrastructure issues (radiod channel
management, payload encoding enforcement, buffer timing, and fusion robustness), but
convergence to UTC(NIST) remains problematic.

### What's Working

1. **Fast Loop** (ring buffer, runs at T=:01):
   - Correctly detects tones and produces D_clock values of +8-12ms
   - Uses 3-second ring buffer centered on minute boundary
   - Updates chrony directly with per-channel results

2. **Slow Loop** (odd/even buffers, runs at T=:06):
   - Now correctly routes samples to capture the minute boundary tone
   - Buffer spans -5s to +55s around minute boundary (60 seconds)
   - Detects tones on many channels with D_clock values of +10-33ms
   - Runs full Phase 2 analysis (BCD correlation, Doppler, 500/600 Hz discrimination)

3. **Buffer Routing** (fixed 2025-12-11):
   - Samples at :55-:59 → next minute's buffer (pre-roll)
   - Samples at :00-:54 → current minute's buffer (contains tone at :00)
   - `system_time = (minute * 60) - 5.0` correctly represents buffer start

4. **Radiod channel management + payload encoding** (fixed 2025-12-12):
   - ka9q-python pinned to 3.2.1
   - Channels are managed by signature (frequency, sample rate, preset, destination)
   - Payload encoding is forced to float32 by setting `OUTPUT_ENCODING=Encoding.F32` via `RadiodControl.tune()`

5. **Fusion robustness** (fixed 2025-12-12):
   - Absurd persisted per-broadcast calibration offsets are clamped/ignored
   - Fusion diagnostics print raw/cal/calibrated values when grade is D or uncertainty is large

### Current Problems to Investigate

1. **Seconds-scale basin errors (`~ -5000ms` and `~ -55000ms`)**: Slow Loop solutions
   still occasionally land in catastrophic basins even with float32 IQ and full-minute
   buffers. Recent logs show Phase2 producing ~-5000ms class D_clock while other
   channels produce tens-of-ms. Fusion then gates (grade D) and/or is pulled.
   Hypotheses to validate:
   - Phase2 minute/second anchoring mismatch (wrong interpretation of `system_time` vs buffer start)
   - Tone-miss fallback behavior that returns `observed=0.00ms` leading to quantized offsets
   - Mixed station discrimination failure on shared frequencies leading to wrong station timebase

2. **Kalman funnel not re-establishing**: Fusion frequently yields grade D and is gated,
   preventing stable convergence. Even when uncertainty is small, grade D gating blocks
   updates. Consider:
   - Update gating should be based on station count + uncertainty, not grade alone
   - Hard reject seconds-scale candidates before fusion/Kalman
   - Explicit Kalman reset/reseed when innovation exceeds threshold

3. **Outlier rejection strategy**: MAD can fail in bimodal distributions (good vs catastrophic).
   Current prefilter of ±100ms is correct for raw D_clock, but we must ensure no later stage
   (calibration, station mapping, minute anchoring) can reintroduce seconds-scale error.

4. **Fast Loop vs Slow Loop discrepancy**: Fast Loop can appear plausible while Slow Loop
   lands in a wrong basin. Both should measure the same D_clock; investigate differences in:
   - tone detector search windows/assumptions
   - minute boundary anchoring
   - station discrimination inputs and how they feed the solver

5. **Calibration clamping**: Fusion robustness fixes have introduced calibration clamping,
   but we need to ensure this doesn't mask underlying issues.

### Key Files to Review

- `src/time_manager/engine/live_time_engine.py`:
  - `ChannelBuffer.add_samples()` - buffer routing logic
  - `_process_slow_loop()` - system_time calculation
  - `_fuse_broadcasts()` - MAD outlier rejection
  - Kalman filter integration

- `src/time_manager/timing/tone_detector.py`:
  - `_detect_tones_internal()` - why does it sometimes return empty?
  - Search window and expected offset parameters

- `src/time_manager/timing/phase2_temporal_engine.py`:
  - `process_minute()` - how system_time is used
  - Step 1 (tone detection) vs Step 3 (solver) timing

- `src/time_manager/timing/clock_convergence.py`:
  - Kalman filter parameters and anomaly detection
  - Should bad measurements be rejected before reaching Kalman?

### Specific Questions

1. Why does Phase2 sometimes converge to a ~-5000ms basin even when tone detection
   claims success and buffers are full-length?

2. Are `system_time`, `start_wallclock`, and the minute boundary calculation consistent
   across Fast Loop, Slow Loop, and Phase2? If not, where is the mismatch introduced?

3. Should we explicitly block seconds-scale candidates before fusion/Kalman and implement
   a hard reset/reseed of convergence state when innovation exceeds a threshold?

4. When Phase2 returns `observed=0.00ms`, is that an explicit failure path that should
   set confidence to ~0 and be rejected before it becomes a D_clock candidate?

5. Why do logs frequently show `start_rtp_wallclock=0.0`? Is radiod timing metadata not
   present in ChannelInfo, or are we not propagating it into the processing path?