"""
Unit tests for Live Time Engine module.

Tests the channel buffer, ring buffer logic, and engine state management.
"""

import pytest
import numpy as np


class TestChannelBuffer:
    """Test ChannelBuffer ring buffer and full buffer logic."""
    
    def test_buffer_initialization(self):
        """Verify buffers are initialized with correct sizes."""
        from time_manager.engine.live_time_engine import ChannelBuffer
        
        buf = ChannelBuffer(
            channel_name="WWV_10_MHz",
            ssrc=12345,
            sample_rate=20000
        )
        
        # Ring buffer: 3 seconds = 60,000 samples
        assert buf.ring_size == 60000
        assert len(buf.ring_buffer) == 60000
        
        # Full buffer: 60 seconds = 1,200,000 samples
        assert buf.full_size == 1200000
        assert len(buf.full_buffer) == 1200000
        
        # Initial positions
        assert buf.ring_write_pos == 0
        assert buf.full_write_pos == 0
    
    def test_full_start_rtp_set_on_first_write(self):
        """Verify full_start_rtp is set on first sample write."""
        from time_manager.engine.live_time_engine import ChannelBuffer
        
        buf = ChannelBuffer(
            channel_name="WWV_10_MHz",
            ssrc=12345
        )
        
        # Initially zero
        assert buf.full_start_rtp == 0
        
        # Add first samples
        samples = np.zeros(1000, dtype=np.complex64)
        buf.add_samples(samples, rtp_timestamp=100000, wallclock=1.0)
        
        # Should be set to first RTP timestamp
        assert buf.full_start_rtp == 100000
        
        # Add more samples - should NOT change
        buf.add_samples(samples, rtp_timestamp=101000, wallclock=2.0)
        assert buf.full_start_rtp == 100000
    
    def test_ring_buffer_wraps_correctly(self):
        """Verify ring buffer wraps around correctly."""
        from time_manager.engine.live_time_engine import ChannelBuffer
        
        buf = ChannelBuffer(
            channel_name="WWV_10_MHz",
            ssrc=12345,
            sample_rate=20000
        )
        
        # Fill ring buffer completely
        chunk_size = 10000
        for i in range(6):  # 6 × 10000 = 60000 = ring_size
            samples = np.ones(chunk_size, dtype=np.complex64) * (i + 1)
            buf.add_samples(samples, rtp_timestamp=i * chunk_size, wallclock=float(i))
        
        assert buf.ring_write_pos == 60000
        
        # Add one more chunk - should wrap
        samples = np.ones(chunk_size, dtype=np.complex64) * 7
        buf.add_samples(samples, rtp_timestamp=60000, wallclock=6.0)
        
        assert buf.ring_write_pos == 70000
        
        # First 10000 samples should now be 7s (overwritten)
        assert buf.ring_buffer[0] == 7
        assert buf.ring_buffer[9999] == 7
        # Samples 10000-19999 should still be 2s
        assert buf.ring_buffer[10000] == 2
    
    def test_reset_clears_positions(self):
        """Verify reset_for_new_minute clears buffer positions."""
        from time_manager.engine.live_time_engine import ChannelBuffer
        
        buf = ChannelBuffer(
            channel_name="WWV_10_MHz",
            ssrc=12345
        )
        
        # Add some samples
        samples = np.zeros(1000, dtype=np.complex64)
        buf.add_samples(samples, rtp_timestamp=100000, wallclock=1.0)
        
        assert buf.full_write_pos > 0
        assert buf.full_start_rtp > 0
        
        # Reset
        buf.reset_for_new_minute()
        
        assert buf.full_write_pos == 0
        assert buf.full_start_rtp == 0
        assert buf.ring_write_pos == 0
        assert buf.full_ready == False


class TestEngineState:
    """Test engine state enumeration."""
    
    def test_engine_states_exist(self):
        """Verify all engine states are defined."""
        from time_manager.engine.live_time_engine import EngineState
        
        assert EngineState.STARTING.value == "STARTING"
        assert EngineState.ACQUIRING.value == "ACQUIRING"
        assert EngineState.TRACKING.value == "TRACKING"
        assert EngineState.HOLDOVER.value == "HOLDOVER"


class TestChannelState:
    """Test per-channel state tracking."""
    
    def test_channel_state_validity(self):
        """Verify channel state validity check."""
        from time_manager.engine.live_time_engine import ChannelState
        
        # Invalid by default
        state = ChannelState()
        assert not state.is_valid()
        
        # Still invalid with partial data
        state.station = "WWV"
        assert not state.is_valid()
        
        state.propagation_mode = "1F"
        assert not state.is_valid()
        
        # Valid with all required fields
        state.propagation_delay_ms = 5.0
        assert state.is_valid()
    
    def test_channel_state_defaults(self):
        """Verify channel state default values."""
        from time_manager.engine.live_time_engine import ChannelState
        
        state = ChannelState()
        
        assert state.station == "UNKNOWN"
        assert state.propagation_mode == "UNKNOWN"
        assert state.n_hops == 1
        assert state.propagation_delay_ms == 0.0
        assert state.snr_db == 0.0
        assert state.confidence == 0.0


class TestLiveTimeEnginePaths:
    """Test LiveTimeEngine configuration."""
    
    def test_state_file_path_is_time_manager(self):
        """Verify state file uses time-manager path, not grape-recorder."""
        from time_manager.engine.live_time_engine import LiveTimeEngine
        
        assert "time-manager" in LiveTimeEngine.STATE_FILE
        assert "grape-recorder" not in LiveTimeEngine.STATE_FILE
    
    def test_shm_path_is_grape_timing(self):
        """Verify SHM path for timing output."""
        from time_manager.engine.live_time_engine import LiveTimeEngine
        
        assert LiveTimeEngine.SHM_PATH == "/dev/shm/grape_timing.json"
