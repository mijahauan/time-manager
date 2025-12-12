"""
Web Server for time-manager GUI.

Provides HTTP endpoints for:
- Static HTML pages (time analysis, discrimination)
- JSON API endpoints for timing and discrimination data
- Server-Sent Events for real-time updates

Extends the health_server.py pattern with additional routes.

Usage:
    from time_manager.web import WebServer
    
    server = WebServer(port=8080)
    server.set_engine(live_time_engine)
    server.start()
"""

import json
import logging
import mimetypes
import os
import threading
import time
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Station azimuths from typical US receiver location (adjustable)
STATION_AZIMUTHS = {
    "WWV": 284,    # Fort Collins, CO (WNW from central US)
    "WWVH": 275,   # Hawaii (W)
    "CHU": 57      # Ottawa, Canada (NE)
}


class WebRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for web GUI endpoints."""
    
    # Class-level references
    engine = None
    static_dir: Optional[Path] = None
    template_dir: Optional[Path] = None
    
    def log_message(self, format, *args):
        """Suppress default HTTP logging for cleaner output."""
        pass
    
    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        
        # Route handling
        if path == '/' or path == '':
            self._redirect('/time')
        elif path == '/time':
            self._serve_template('time.html')
        elif path == '/discrimination':
            self._serve_template('discrimination.html')
        elif path == '/health':
            self._handle_health()
        elif path == '/status':
            self._handle_status()
        elif path == '/metrics':
            self._handle_metrics()
        # API endpoints
        elif path == '/api/timing':
            self._handle_api_timing()
        elif path == '/api/timing/history':
            minutes = int(query.get('minutes', [60])[0])
            self._handle_api_timing_history(minutes)
        elif path == '/api/timing/fusion':
            self._handle_api_timing_fusion()
        elif path == '/api/timing/constellation':
            self._handle_api_timing_constellation()
        elif path == '/api/timing/consensus':
            self._handle_api_timing_consensus()
        elif path == '/api/timing/mode-probability':
            channel = query.get('channel', [None])[0]
            self._handle_api_mode_probability(channel)
        elif path == '/api/discrimination':
            self._handle_api_discrimination_summary()
        elif path.startswith('/api/discrimination/'):
            # /api/discrimination/{channel}
            channel = path.split('/api/discrimination/')[1]
            self._handle_api_discrimination_channel(channel)
        elif path == '/api/channels':
            self._handle_api_channels()
        elif path == '/api/chrony':
            self._handle_api_chrony()
        elif path == '/api/chrony/history':
            minutes = int(query.get('minutes', [60])[0])
            self._handle_api_chrony_history(minutes)
        elif path == '/events':
            self._handle_sse()
        # Static files
        elif path.startswith('/static/'):
            self._serve_static(path[8:])  # Remove '/static/' prefix
        else:
            self.send_error(404, "Not Found")
    
    def _redirect(self, location: str):
        """Send HTTP redirect."""
        self.send_response(302)
        self.send_header('Location', location)
        self.end_headers()
    
    def _send_json(self, data: Any, status: int = 200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2, cls=NumpyEncoder).encode())
    
    def _send_html(self, content: str, status: int = 200):
        """Send HTML response."""
        self.send_response(status)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(content.encode())
    
    def _serve_template(self, template_name: str):
        """Serve an HTML template."""
        if not self.template_dir:
            self.send_error(500, "Template directory not configured")
            return
        
        template_path = self.template_dir / template_name
        if not template_path.exists():
            self.send_error(404, f"Template not found: {template_name}")
            return
        
        try:
            content = template_path.read_text()
            self._send_html(content)
        except Exception as e:
            self.send_error(500, f"Error reading template: {e}")
    
    def _serve_static(self, file_path: str):
        """Serve a static file."""
        if not self.static_dir:
            self.send_error(500, "Static directory not configured")
            return
        
        # Security: prevent directory traversal by checking for ..
        if '..' in file_path:
            self.send_error(403, "Forbidden")
            return
        
        full_path = self.static_dir / file_path
        
        # Verify resolved path is within static_dir
        try:
            full_path.resolve().relative_to(self.static_dir.resolve())
        except ValueError:
            self.send_error(403, "Forbidden")
            return
        
        if not full_path.exists():
            self.send_error(404, f"File not found: {file_path}")
            return
        
        try:
            content = full_path.read_bytes()
            mime_type, _ = mimetypes.guess_type(str(full_path))
            
            self.send_response(200)
            self.send_header('Content-Type', mime_type or 'application/octet-stream')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")
    
    # =========================================================================
    # Health endpoints (from health_server.py)
    # =========================================================================
    
    def _handle_health(self):
        """Basic health check."""
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'OK\n')
    
    def _handle_status(self):
        """JSON status with timing information."""
        if not self.engine:
            self._send_json({'error': 'No engine connected'}, 503)
            return
        
        try:
            status = self._get_engine_status()
            self._send_json(status)
        except Exception as e:
            self._send_json({'error': str(e)}, 500)
    
    def _handle_metrics(self):
        """Prometheus-compatible metrics."""
        if not self.engine:
            self.send_response(503)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'# No engine connected\n')
            return
        
        try:
            status = self._get_engine_status()
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
    
    def _get_engine_status(self) -> Dict[str, Any]:
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
        
        # Per-channel status
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
        
        # Last fusion result
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
        
        return status
    
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
        ]
        lines.append('')
        return '\n'.join(lines)
    
    # =========================================================================
    # Timing API endpoints
    # =========================================================================
    
    def _handle_api_timing(self):
        """Current timing state."""
        if not self.engine:
            self._send_json({'error': 'No engine connected'}, 503)
            return
        
        data = {
            'timestamp': time.time(),
            'd_clock_ms': self.engine.d_clock_ms,
            'd_clock_uncertainty_ms': self.engine.d_clock_uncertainty_ms,
            'clock_status': self.engine.state.value if hasattr(self.engine.state, 'value') else 'UNKNOWN',
            'quality_grade': 'B',  # Default
        }
        
        # Add fusion data
        if hasattr(self.engine, 'last_fusion') and self.engine.last_fusion:
            fusion = self.engine.last_fusion
            data['quality_grade'] = fusion.quality_grade
            data['fusion'] = {
                'n_broadcasts': fusion.n_broadcasts,
                'wwv_count': fusion.wwv_count,
                'wwvh_count': fusion.wwvh_count,
                'chu_count': fusion.chu_count,
                'outliers_rejected': fusion.n_outliers_rejected
            }
        
        # Add per-channel data (include all channels with station identified)
        if hasattr(self.engine, 'channel_states'):
            data['channels'] = {}
            for name, state in self.engine.channel_states.items():
                # Include channel if station has been identified (not UNKNOWN)
                if state.station != "UNKNOWN":
                    data['channels'][name] = {
                        'station': state.station,
                        'd_clock_raw_ms': getattr(state, 'd_clock_raw_ms', 0.0),
                        'propagation_delay_ms': state.propagation_delay_ms,
                        'propagation_mode': state.propagation_mode,
                        'snr_db': state.snr_db,
                        'confidence': state.confidence,
                        'valid': state.is_valid()
                    }
        
        self._send_json(data)
    
    def _handle_api_timing_history(self, minutes: int = 60):
        """Historical D_clock for Kalman funnel chart."""
        if not self.engine:
            self._send_json({'error': 'No engine connected'}, 503)
            return
        
        # Get from engine's fusion history
        history = list(self.engine.fusion_history) if hasattr(self.engine, 'fusion_history') else []
        
        # Filter to requested time window
        cutoff = time.time() - (minutes * 60)
        filtered = [h for h in history if h.get('timestamp', 0) >= cutoff]
        
        self._send_json({
            'minutes_requested': minutes,
            'points': len(filtered),
            'history': filtered
        })
    
    def _handle_api_timing_fusion(self):
        """Fusion calibration and history for charts."""
        if not self.engine:
            self._send_json({'error': 'No engine connected'}, 503)
            return
        
        data = {
            'status': 'active' if self.engine.last_fusion else 'inactive',
            'calibration': {},
            'history': []
        }
        
        # Add calibration offsets
        if hasattr(self.engine, 'calibration'):
            for key, cal in self.engine.calibration.items():
                data['calibration'][cal.station] = {
                    'offset_ms': cal.offset_ms,
                    'uncertainty_ms': cal.uncertainty_ms,
                    'n_samples': cal.n_samples
                }
        
        # Add fusion history from engine
        if hasattr(self.engine, 'fusion_history'):
            data['history'] = list(self.engine.fusion_history)
        
        self._send_json(data)
    
    def _handle_api_timing_constellation(self):
        """Station errors by azimuth for constellation radar."""
        if not self.engine:
            self._send_json({'error': 'No engine connected'}, 503)
            return
        
        stations = []
        
        if hasattr(self.engine, 'channel_states'):
            for name, state in self.engine.channel_states.items():
                if state.is_valid():
                    base_station = state.station
                    azimuth = STATION_AZIMUTHS.get(base_station, 0)
                    
                    # Calculate error (D_clock from this channel)
                    error_ms = getattr(state, 'd_clock_raw_ms', 0.0)
                    
                    # Apply calibration if available
                    if hasattr(self.engine, 'calibration'):
                        for key, cal in self.engine.calibration.items():
                            if cal.station == base_station:
                                error_ms += cal.offset_ms
                                break
                    
                    stations.append({
                        'name': name,
                        'channel': name,
                        'base_station': base_station,
                        'azimuth_deg': azimuth,
                        'error_ms': error_ms,
                        'snr': state.snr_db,
                        'active': True
                    })
        
        self._send_json({'stations': stations})
    
    def _handle_api_timing_consensus(self):
        """Consensus estimates for probability peak KDE."""
        if not self.engine:
            self._send_json({'error': 'No engine connected'}, 503)
            return
        
        estimates = []
        
        if hasattr(self.engine, 'channel_states'):
            for name, state in self.engine.channel_states.items():
                if state.is_valid():
                    offset = getattr(state, 'd_clock_raw_ms', 0.0)
                    
                    # Apply calibration
                    if hasattr(self.engine, 'calibration'):
                        for key, cal in self.engine.calibration.items():
                            if cal.station == state.station:
                                offset += cal.offset_ms
                                break
                    
                    estimates.append({
                        'source': name,
                        'offset': offset,
                        'uncertainty': getattr(state, 'uncertainty_ms', 2.0),
                        'station': state.station,
                        'weight': state.confidence
                    })
        
        self._send_json({'estimates': estimates})
    
    def _handle_api_mode_probability(self, channel: Optional[str]):
        """Propagation mode probability for ridge chart."""
        if not self.engine:
            self._send_json({'error': 'No engine connected'}, 503)
            return
        
        # Find best channel if not specified
        if not channel and hasattr(self.engine, 'channel_states'):
            # Pick channel with highest SNR
            best_snr = -999
            for name, state in self.engine.channel_states.items():
                if state.is_valid() and state.snr_db > best_snr:
                    best_snr = state.snr_db
                    channel = name
        
        if not channel:
            self._send_json({'available': False, 'error': 'No channel available'})
            return
        
        # Get mode probability from transmission time solver if available
        candidates = []
        measured_delay = None
        
        if hasattr(self.engine, 'channel_states') and channel in self.engine.channel_states:
            state = self.engine.channel_states[channel]
            measured_delay = state.propagation_delay_ms
            
            # Generate candidate modes based on typical propagation
            # These would ideally come from TransmissionTimeSolver
            mode_delays = {
                '1F': measured_delay * 0.7 if measured_delay else 5.0,
                '2F': measured_delay if measured_delay else 8.0,
                '3F': measured_delay * 1.3 if measured_delay else 12.0,
            }
            
            # Assign probabilities based on current mode
            current_mode = state.propagation_mode
            for mode, delay in mode_delays.items():
                prob = 0.8 if mode == current_mode else 0.1
                candidates.append({
                    'mode': mode,
                    'delay_ms': delay,
                    'probability': prob
                })
        
        self._send_json({
            'available': True,
            'channel': channel,
            'candidates': candidates,
            'measured_delay': measured_delay
        })
    
    # =========================================================================
    # Discrimination API endpoints
    # =========================================================================
    
    def _handle_api_discrimination_summary(self):
        """Summary of discrimination across all channels."""
        if not self.engine:
            self._send_json({'error': 'No engine connected'}, 503)
            return
        
        channels = {}
        
        # Get discrimination data from Phase2 engines (discriminators are inside them)
        if hasattr(self.engine, '_phase2_engines'):
            for name, phase2_engine in self.engine._phase2_engines.items():
                if hasattr(phase2_engine, 'discriminator'):
                    disc = phase2_engine.discriminator
                    if hasattr(disc, 'measurements') and disc.measurements:
                        latest = disc.measurements[-1]
                        channels[name] = {
                            'dominant_station': latest.dominant_station,
                            'confidence': latest.confidence,
                            'power_ratio_db': getattr(latest, 'power_ratio_db', 0.0),
                            'wwv_detected': getattr(latest, 'wwv_detected', False),
                            'wwvh_detected': getattr(latest, 'wwvh_detected', False),
                            'measurement_count': len(disc.measurements)
                        }
        
        self._send_json({
            'timestamp': time.time(),
            'channels': channels
        })
    
    def _handle_api_discrimination_channel(self, channel: str):
        """Detailed discrimination data for a specific channel."""
        if not self.engine:
            self._send_json({'error': 'No engine connected'}, 503)
            return
        
        # URL decode channel name
        from urllib.parse import unquote
        channel = unquote(channel)
        
        if not hasattr(self.engine, '_phase2_engines'):
            self._send_json({'error': 'No Phase2 engines available'}, 404)
            return
        
        phase2_engine = self.engine._phase2_engines.get(channel)
        if not phase2_engine or not hasattr(phase2_engine, 'discriminator'):
            self._send_json({'error': f'Channel not found: {channel}'}, 404)
            return
        
        disc = phase2_engine.discriminator
        
        # Get measurements (last 1440 = 24 hours)
        measurements = disc.measurements[-1440:] if hasattr(disc, 'measurements') else []
        
        timeline = []
        for m in measurements:
            entry = {
                'timestamp_utc': m.minute_timestamp,
                'wwv_detected': m.wwv_detected,
                'wwvh_detected': m.wwvh_detected,
                'wwv_power_db': m.wwv_power_db,
                'wwvh_power_db': m.wwvh_power_db,
                'power_ratio_db': m.power_ratio_db,
                'differential_delay_ms': m.differential_delay_ms,
                'dominant_station': m.dominant_station,
                'confidence': m.confidence,
            }
            
            # Add method-specific data if available
            if hasattr(m, 'bcd_wwv_amplitude'):
                entry['bcd_wwv_amplitude'] = m.bcd_wwv_amplitude
                entry['bcd_wwvh_amplitude'] = m.bcd_wwvh_amplitude
                entry['bcd_correlation_quality'] = m.bcd_correlation_quality
            
            if hasattr(m, 'tone_440hz_wwv_detected'):
                entry['tone_440hz_wwv_detected'] = m.tone_440hz_wwv_detected
                entry['tone_440hz_wwvh_detected'] = m.tone_440hz_wwvh_detected
            
            if hasattr(m, 'test_signal_detected'):
                entry['test_signal_detected'] = m.test_signal_detected
                entry['test_signal_station'] = m.test_signal_station
            
            if hasattr(m, 'doppler_wwv_hz'):
                entry['doppler_wwv_hz'] = m.doppler_wwv_hz
                entry['doppler_wwvh_hz'] = m.doppler_wwvh_hz
            
            timeline.append(entry)
        
        # Calculate summary statistics
        wwv_count = sum(1 for m in measurements if m.dominant_station == 'WWV')
        wwvh_count = sum(1 for m in measurements if m.dominant_station == 'WWVH')
        balanced_count = sum(1 for m in measurements if m.dominant_station == 'BALANCED')
        total = len(measurements)
        
        self._send_json({
            'channel': channel,
            'summary': {
                'total_minutes': total,
                'wwv_dominant': wwv_count,
                'wwvh_dominant': wwvh_count,
                'balanced': balanced_count,
                'dominance_pct': {
                    'wwv': round(100 * wwv_count / total, 1) if total > 0 else 0,
                    'wwvh': round(100 * wwvh_count / total, 1) if total > 0 else 0,
                    'balanced': round(100 * balanced_count / total, 1) if total > 0 else 0
                }
            },
            'timeline': timeline
        })
    
    def _handle_api_channels(self):
        """List all available channels."""
        if not self.engine:
            self._send_json({'error': 'No engine connected'}, 503)
            return
        
        channels = []
        
        if hasattr(self.engine, 'channel_states'):
            for name, state in self.engine.channel_states.items():
                channels.append({
                    'name': name,
                    'station': state.station,
                    'valid': state.is_valid(),
                    'snr_db': state.snr_db,
                    'propagation_mode': state.propagation_mode
                })
        
        self._send_json({'channels': channels})
    
    # =========================================================================
    # Chrony API endpoints
    # =========================================================================
    
    def _handle_api_chrony(self):
        """Current chrony sources status."""
        import subprocess
        
        try:
            result = subprocess.run(
                ['chronyc', '-c', 'sources'],
                capture_output=True, text=True, timeout=5
            )
            
            sources = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) >= 10:
                    # Format: mode,state,name,stratum,poll,reach,lastRx,offset,error,uncertainty
                    source = {
                        'mode': parts[0],  # ^ = server, # = local refclock
                        'state': parts[1],  # * = current, + = combined, - = not combined
                        'name': parts[2],
                        'stratum': int(parts[3]) if parts[3].isdigit() else 0,
                        'poll': int(parts[4]) if parts[4].lstrip('-').isdigit() else 0,
                        'reach': int(parts[5]) if parts[5].isdigit() else 0,
                        'last_rx': int(parts[6]) if parts[6].lstrip('-').isdigit() else 0,
                        'offset_s': float(parts[7]) if parts[7] else 0.0,
                        'offset_ms': float(parts[7]) * 1000 if parts[7] else 0.0,
                        'error_s': float(parts[8]) if parts[8] else 0.0,
                        'uncertainty_s': float(parts[9]) if parts[9] else 0.0
                    }
                    sources.append(source)
            
            # Find TMGR and GPS server specifically
            tmgr = next((s for s in sources if s['name'] == 'TMGR'), None)
            gps_server = next((s for s in sources if s['name'] == '192.168.0.134'), None)
            
            # Find best NTP server (excluding TMGR and local GPS)
            ntp_sources = [s for s in sources if s['mode'] == '^' and s['name'] != '192.168.0.134']
            best_ntp = min(ntp_sources, key=lambda s: abs(s['offset_s'])) if ntp_sources else None
            
            self._send_json({
                'timestamp': time.time(),
                'sources': sources,
                'tmgr': tmgr,
                'gps_server': gps_server,
                'best_ntp': best_ntp
            })
            
        except subprocess.TimeoutExpired:
            self._send_json({'error': 'chronyc timeout'}, 500)
        except FileNotFoundError:
            self._send_json({'error': 'chronyc not found'}, 500)
        except Exception as e:
            self._send_json({'error': str(e)}, 500)
    
    def _handle_api_chrony_history(self, minutes: int = 60):
        """Historical chrony data from engine's chrony_history buffer."""
        if not self.engine:
            self._send_json({'error': 'No engine connected'}, 503)
            return
        
        history = list(self.engine.chrony_history) if hasattr(self.engine, 'chrony_history') else []
        
        # Filter to requested time window
        cutoff = time.time() - (minutes * 60)
        filtered = [h for h in history if h.get('timestamp', 0) >= cutoff]
        
        self._send_json({
            'minutes_requested': minutes,
            'points': len(filtered),
            'history': filtered
        })
    
    # =========================================================================
    # Server-Sent Events
    # =========================================================================
    
    def _handle_sse(self):
        """Server-Sent Events endpoint for real-time updates."""
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        try:
            while True:
                if self.engine:
                    # Send timing update
                    data = {
                        'type': 'timing',
                        'timestamp': time.time(),
                        'd_clock_ms': self.engine.d_clock_ms,
                        'd_clock_uncertainty_ms': self.engine.d_clock_uncertainty_ms,
                        'state': self.engine.state.value if hasattr(self.engine.state, 'value') else 'UNKNOWN'
                    }
                    
                    self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
                    self.wfile.flush()
                
                time.sleep(5)  # Update every 5 seconds
                
        except (BrokenPipeError, ConnectionResetError):
            pass  # Client disconnected


class WebServer:
    """
    HTTP server for time-manager web GUI.
    
    Runs in a background thread and provides:
    - Time analysis page with Kalman funnel, constellation radar, probability peak
    - Discrimination page with 13-method voting visualization
    - JSON API endpoints for data
    - Server-Sent Events for real-time updates
    """
    
    def __init__(self, port: int = 8080, bind_address: str = '0.0.0.0'):
        """
        Initialize the web server.
        
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
        
        # Set up paths for static files and templates
        self.web_dir = Path(__file__).parent
        self.static_dir = self.web_dir / 'static'
        self.template_dir = self.web_dir / 'templates'
        
        # Configure handler class
        WebRequestHandler.static_dir = self.static_dir
        WebRequestHandler.template_dir = self.template_dir
    
    def set_engine(self, engine):
        """
        Connect to a LiveTimeEngine for data.
        
        Args:
            engine: LiveTimeEngine instance
        """
        self.engine = engine
        WebRequestHandler.engine = engine
        
        # Register callback to update fusion history
        if hasattr(engine, 'on_fusion_update'):
            engine.on_fusion_update = self._on_fusion_update
    
    def _on_fusion_update(self, fusion_result):
        """Callback when fusion result is updated (for SSE notifications)."""
        # Engine now maintains its own fusion_history
        # This callback can be used for SSE push notifications if needed
        pass
    
    def start(self):
        """Start the web server in a background thread."""
        if self._running:
            logger.warning("Web server already running")
            return
        
        try:
            # Use ThreadingMixIn for concurrent request handling
            class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
                daemon_threads = True
            
            self.server = ThreadedHTTPServer(
                (self.bind_address, self.port),
                WebRequestHandler
            )
            self._running = True
            
            self.thread = threading.Thread(
                target=self._serve,
                name="WebServer",
                daemon=True
            )
            self.thread.start()
            
            logger.info(f"Web server started on http://{self.bind_address}:{self.port}")
            logger.info(f"  GET /time           - Time analysis page")
            logger.info(f"  GET /discrimination - Discrimination page")
            logger.info(f"  GET /api/timing     - Timing API")
            logger.info(f"  GET /events         - Server-Sent Events")
            
        except Exception as e:
            logger.error(f"Failed to start web server: {e}")
            self._running = False
    
    def _serve(self):
        """Server loop (runs in background thread)."""
        self.server.serve_forever()
    
    def stop(self):
        """Stop the web server."""
        self._running = False
        if self.server:
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception:
                pass
            self.server = None
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("Web server stopped")
