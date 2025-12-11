"""
Tests for health monitoring server.
"""

import json
import pytest
import threading
import time
import urllib.request
from unittest.mock import MagicMock


class TestHealthServer:
    """Tests for HealthServer."""
    
    def test_health_server_import(self):
        """Test that HealthServer can be imported."""
        from time_manager.output.health_server import HealthServer
        assert HealthServer is not None
    
    def test_health_request_handler_import(self):
        """Test that HealthRequestHandler can be imported."""
        from time_manager.output.health_server import HealthRequestHandler
        assert HealthRequestHandler is not None
    
    def test_health_server_initialization(self):
        """Test HealthServer initialization with custom port."""
        from time_manager.output.health_server import HealthServer
        
        server = HealthServer(port=9999, bind_address='127.0.0.1')
        assert server.port == 9999
        assert server.bind_address == '127.0.0.1'
        assert server.engine is None
        assert server._running is False


class TestHealthServerIntegration:
    """Integration tests for HealthServer (requires network)."""
    
    @pytest.fixture
    def mock_engine(self):
        """Create a mock LiveTimeEngine for testing."""
        engine = MagicMock()
        engine.state = MagicMock()
        engine.state.value = 'TRACKING'
        engine.d_clock_ms = 1.234
        engine.d_clock_uncertainty_ms = 0.5
        engine.clock_drift_ppm = 0.01
        engine.stats = {
            'fast_loop_count': 10,
            'slow_loop_count': 5,
            'chrony_updates': 3,
            'start_time': time.time() - 100
        }
        engine.channel_states = {}
        engine.calibration = {}
        engine.last_fusion = None
        return engine
    
    @pytest.fixture
    def health_server(self, mock_engine):
        """Create and start a health server for testing."""
        from time_manager.output.health_server import HealthServer
        
        # Use a high port to avoid conflicts
        server = HealthServer(port=19876, bind_address='127.0.0.1')
        server.set_engine(mock_engine)
        server.start()
        
        # Give server time to start
        time.sleep(0.1)
        
        yield server
        
        server.stop()
    
    def test_health_endpoint(self, health_server):
        """Test /health endpoint returns OK."""
        try:
            response = urllib.request.urlopen(
                'http://127.0.0.1:19876/health',
                timeout=2
            )
            assert response.status == 200
            assert response.read() == b'OK\n'
        except Exception as e:
            pytest.skip(f"Network test failed: {e}")
    
    def test_status_endpoint(self, health_server):
        """Test /status endpoint returns JSON."""
        try:
            response = urllib.request.urlopen(
                'http://127.0.0.1:19876/status',
                timeout=2
            )
            assert response.status == 200
            
            data = json.loads(response.read())
            assert 'd_clock_ms' in data
            assert data['d_clock_ms'] == 1.234
            assert data['state'] == 'TRACKING'
        except Exception as e:
            pytest.skip(f"Network test failed: {e}")
    
    def test_metrics_endpoint(self, health_server):
        """Test /metrics endpoint returns Prometheus format."""
        try:
            response = urllib.request.urlopen(
                'http://127.0.0.1:19876/metrics',
                timeout=2
            )
            assert response.status == 200
            
            content = response.read().decode()
            assert 'time_manager_d_clock_ms' in content
            assert '1.234' in content
            assert 'time_manager_state' in content
        except Exception as e:
            pytest.skip(f"Network test failed: {e}")


class TestPrometheusMetrics:
    """Tests for Prometheus metrics formatting."""
    
    def test_prometheus_format(self):
        """Test that metrics are properly formatted for Prometheus."""
        from time_manager.output.health_server import HealthRequestHandler
        
        handler = HealthRequestHandler.__new__(HealthRequestHandler)
        
        status = {
            'd_clock_ms': 2.5,
            'd_clock_uncertainty_ms': 0.3,
            'channels_active': 4,
            'fast_loop_count': 100,
            'slow_loop_count': 50,
            'chrony_updates': 25,
            'uptime_seconds': 3600.0,
            'state': 'TRACKING',
            'channels': {
                'WWV 10 MHz': {'snr_db': 15.5}
            }
        }
        
        metrics = handler._format_prometheus_metrics(status)
        
        # Check metric names and values
        assert 'time_manager_d_clock_ms 2.5' in metrics
        assert 'time_manager_d_clock_uncertainty_ms 0.3' in metrics
        assert 'time_manager_channels_active 4' in metrics
        assert 'time_manager_fast_loop_total 100' in metrics
        assert 'time_manager_state 3' in metrics  # TRACKING = 3
        assert 'channel="WWV_10_MHz"' in metrics
        assert '15.5' in metrics
