"""
Health Monitoring HTTP Server for time-manager.

Provides a simple HTTP endpoint for monitoring timing status, metrics,
and health checks. Useful for integration with monitoring systems like
Prometheus, Grafana, or simple health checks.

Endpoints:
    GET /health     - Basic health check (200 OK if running)
    GET /status     - JSON timing status and metrics
    GET /metrics    - Prometheus-compatible metrics

Usage:
    from time_manager.output.health_server import HealthServer
    
    server = HealthServer(port=8080)
    server.set_engine(live_time_engine)
    server.start()
"""

import json
import logging
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class HealthRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for health endpoints."""
    
    # Class-level reference to status callback
    get_status: Optional[Callable[[], Dict[str, Any]]] = None
    
    def log_message(self, format, *args):
        """Suppress default HTTP logging."""
        pass
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/health':
            self._handle_health()
        elif self.path == '/status':
            self._handle_status()
        elif self.path == '/metrics':
            self._handle_metrics()
        else:
            self.send_error(404, "Not Found")
    
    def _handle_health(self):
        """Basic health check - returns 200 if server is running."""
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'OK\n')
    
    def _handle_status(self):
        """Return JSON status with timing information."""
        if self.get_status:
            try:
                status = self.get_status()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(status, indent=2).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())
        else:
            self.send_response(503)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'No engine connected'}).encode())
    
    def _handle_metrics(self):
        """Return Prometheus-compatible metrics."""
        if self.get_status:
            try:
                status = self.get_status()
                metrics = self._format_prometheus_metrics(status)
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain; version=0.0.4')
                self.end_headers()
                self.wfile.write(metrics.encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write(f'# Error: {e}\n'.encode())
        else:
            self.send_response(503)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'# No engine connected\n')
    
    def _format_prometheus_metrics(self, status: Dict[str, Any]) -> str:
        """Format status as Prometheus metrics."""
        lines = [
            '# HELP time_manager_d_clock_ms System clock offset from UTC(NIST) in milliseconds',
            '# TYPE time_manager_d_clock_ms gauge',
            f'time_manager_d_clock_ms {status.get("d_clock_ms", 0):.6f}',
            '',
            '# HELP time_manager_d_clock_uncertainty_ms Uncertainty of clock offset in milliseconds',
            '# TYPE time_manager_d_clock_uncertainty_ms gauge',
            f'time_manager_d_clock_uncertainty_ms {status.get("d_clock_uncertainty_ms", 0):.6f}',
            '',
            '# HELP time_manager_channels_active Number of active timing channels',
            '# TYPE time_manager_channels_active gauge',
            f'time_manager_channels_active {status.get("channels_active", 0)}',
            '',
            '# HELP time_manager_fast_loop_total Total fast loop iterations',
            '# TYPE time_manager_fast_loop_total counter',
            f'time_manager_fast_loop_total {status.get("fast_loop_count", 0)}',
            '',
            '# HELP time_manager_slow_loop_total Total slow loop iterations',
            '# TYPE time_manager_slow_loop_total counter',
            f'time_manager_slow_loop_total {status.get("slow_loop_count", 0)}',
            '',
            '# HELP time_manager_chrony_updates_total Total Chrony SHM updates',
            '# TYPE time_manager_chrony_updates_total counter',
            f'time_manager_chrony_updates_total {status.get("chrony_updates", 0)}',
            '',
            '# HELP time_manager_uptime_seconds Engine uptime in seconds',
            '# TYPE time_manager_uptime_seconds gauge',
            f'time_manager_uptime_seconds {status.get("uptime_seconds", 0):.1f}',
            '',
            '# HELP time_manager_state Engine state (1=STARTING, 2=ACQUIRING, 3=TRACKING, 4=HOLDOVER)',
            '# TYPE time_manager_state gauge',
        ]
        
        state_map = {'STARTING': 1, 'ACQUIRING': 2, 'TRACKING': 3, 'HOLDOVER': 4}
        state_value = state_map.get(status.get('state', 'STARTING'), 0)
        lines.append(f'time_manager_state {state_value}')
        
        # Per-channel metrics
        channels = status.get('channels', {})
        if channels:
            lines.extend([
                '',
                '# HELP time_manager_channel_snr_db Channel signal-to-noise ratio in dB',
                '# TYPE time_manager_channel_snr_db gauge',
            ])
            for name, ch_status in channels.items():
                snr = ch_status.get('snr_db', 0)
                safe_name = name.replace(' ', '_').replace('.', '_')
                lines.append(f'time_manager_channel_snr_db{{channel="{safe_name}"}} {snr:.1f}')
        
        lines.append('')
        return '\n'.join(lines)


class HealthServer:
    """
    HTTP server for health monitoring.
    
    Runs in a background thread and provides endpoints for monitoring
    the time-manager daemon.
    """
    
    def __init__(self, port: int = 8080, bind_address: str = '0.0.0.0'):
        """
        Initialize the health server.
        
        Args:
            port: HTTP port to listen on
            bind_address: Address to bind to (default: all interfaces)
        """
        self.port = port
        self.bind_address = bind_address
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.engine = None
        self._running = False
    
    def set_engine(self, engine):
        """
        Connect to a LiveTimeEngine for status reporting.
        
        Args:
            engine: LiveTimeEngine instance
        """
        self.engine = engine
        HealthRequestHandler.get_status = self._get_status
    
    def _get_status(self) -> Dict[str, Any]:
        """Get current status from the engine."""
        if not self.engine:
            return {'error': 'No engine connected'}
        
        status = {
            'timestamp': time.time(),
            'state': self.engine.state.value if hasattr(self.engine.state, 'value') else str(self.engine.state),
            'd_clock_ms': self.engine.d_clock_ms,
            'd_clock_uncertainty_ms': self.engine.d_clock_uncertainty_ms,
            'clock_drift_ppm': self.engine.clock_drift_ppm,
            'fast_loop_count': self.engine.stats.get('fast_loop_count', 0),
            'slow_loop_count': self.engine.stats.get('slow_loop_count', 0),
            'chrony_updates': self.engine.stats.get('chrony_updates', 0),
            'uptime_seconds': time.time() - self.engine.stats.get('start_time', time.time()),
            'channels_active': len([
                name for name, state in self.engine.channel_states.items()
                if state.is_valid()
            ]) if hasattr(self.engine, 'channel_states') else 0,
        }
        
        # Add per-channel status
        if hasattr(self.engine, 'channel_states'):
            status['channels'] = {}
            for name, state in self.engine.channel_states.items():
                status['channels'][name] = {
                    'station': state.station,
                    'propagation_mode': state.propagation_mode,
                    'propagation_delay_ms': state.propagation_delay_ms,
                    'snr_db': state.snr_db,
                    'confidence': state.confidence,
                    'valid': state.is_valid()
                }
        
        # Add last fusion result if available
        if hasattr(self.engine, 'last_fusion') and self.engine.last_fusion:
            fusion = self.engine.last_fusion
            status['last_fusion'] = {
                'd_clock_ms': fusion.d_clock_ms,
                'uncertainty_ms': fusion.uncertainty_ms,
                'n_broadcasts': fusion.n_broadcasts,
                'quality_grade': fusion.quality_grade,
                'wwv_count': fusion.wwv_count,
                'wwvh_count': fusion.wwvh_count,
                'chu_count': fusion.chu_count
            }
        
        # Add calibration summary
        if hasattr(self.engine, 'calibration'):
            status['calibration_count'] = len(self.engine.calibration)
            status['calibrations'] = {
                key: {
                    'offset_ms': cal.offset_ms,
                    'uncertainty_ms': cal.uncertainty_ms,
                    'n_samples': cal.n_samples
                }
                for key, cal in self.engine.calibration.items()
            }
        
        return status
    
    def start(self):
        """Start the health server in a background thread."""
        if self._running:
            logger.warning("Health server already running")
            return
        
        try:
            self.server = HTTPServer(
                (self.bind_address, self.port),
                HealthRequestHandler
            )
            # Set timeout so handle_request doesn't block forever
            self.server.timeout = 1.0
            self._running = True
            
            self.thread = threading.Thread(
                target=self._serve,
                name="HealthServer",
                daemon=True
            )
            self.thread.start()
            
            logger.info(f"Health server started on http://{self.bind_address}:{self.port}")
            logger.info(f"  GET /health  - Health check")
            logger.info(f"  GET /status  - JSON status")
            logger.info(f"  GET /metrics - Prometheus metrics")
            
        except Exception as e:
            logger.error(f"Failed to start health server: {e}")
            self._running = False
    
    def _serve(self):
        """Server loop (runs in background thread)."""
        while self._running:
            try:
                self.server.handle_request()
            except Exception:
                pass  # Timeout or shutdown
    
    def stop(self):
        """Stop the health server."""
        self._running = False
        if self.server:
            try:
                self.server.server_close()
            except Exception:
                pass
            self.server = None
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("Health server stopped")
