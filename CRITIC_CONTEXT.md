# NEVER CHANGE THE FOLLOWING PRIMARY INSTRUCTION:

Primary Instruction:  In this context you will perform a critical review of the GRAPE Recorder project, either in its entirety or in a specific component, as specified by the user.  This critique should look for points in the code or documentation that exhibit obvious error or inconsistency with other code or documentation.  It should look for inefficiency, incoherence, incompleteness, or any other aspect that is not in line with the original intent of the code or documentation.  It should also look for obsolete, deprecated, or "zombie" code that should be removed.  Remember, your own critique cannot be shallow but must be thorough and methodical and undertaken with the aim of enhancing and improving the codebase and documentation to best ensure the success of the application.

# The following secondary instruction and information will guide your critique in this particular session (the instructions below will vary from session to session):

## Session Focus: Convergence Methodology Issues

The LiveTimeEngine has been debugged to the point where both Fast Loop and Slow Loop
are producing valid D_clock measurements, but convergence remains problematic.

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

### Current Problems to Investigate

1. **Intermittent `observed=0.00ms`**: Some channels in the Slow Loop still produce
   `observed=0.00ms` which causes 55-second D_clock offsets. This happens even when
   the Fast Loop detects tones on the same channel. Possible causes:
   - Phase 2 tone detector not finding tone (different search parameters?)
   - Buffer timing still off for some edge cases
   - Sample gaps or discontinuities in the buffer

2. **Kalman Filter Corruption**: When bad measurements (55s offset) are fed to the
   Kalman filter, it takes a long time to recover. The innovation values are huge
   (e.g., `innov=+41806.93ms`). Consider:
   - Pre-filtering measurements before Kalman (reject >1 second offsets?)
   - Kalman reset logic when innovation exceeds threshold
   - Separate Kalman instances for Fast vs Slow Loop

3. **MAD Outlier Rejection Ineffective**: The MAD-based outlier rejection doesn't
   work well when there's a mix of good (+10ms) and bad (+55000ms) measurements.
   The MAD becomes huge, so nothing gets rejected.

4. **Fast Loop vs Slow Loop Discrepancy**: Fast Loop produces consistent +8-12ms
   values, but Slow Loop produces a wider range (+10-33ms) with some 55s outliers.
   Why the difference? Both should be measuring the same clock offset.

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

1. Why does the Slow Loop Phase 2 tone detector sometimes fail to find tones that
   the Fast Loop successfully detects on the same channel?

2. Is the `system_time = (minute * 60) - 5.0` calculation correct for all cases,
   or are there edge cases where the buffer doesn't actually start at :55?

3. Should the Kalman filter have a "reset" mechanism when innovation exceeds a
   threshold (e.g., >1 second)?

4. Is there a race condition between buffer filling and buffer reading that could
   cause incomplete or corrupted data?

5. The Fast Loop uses `buffer.last_wallclock` for timing while Slow Loop uses
   a calculated `system_time`. Should they use the same approach?