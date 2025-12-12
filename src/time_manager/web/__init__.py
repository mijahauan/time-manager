"""
Web GUI module for time-manager.

Provides HTTP server with:
- Time analysis page (Kalman funnel, constellation radar, probability peak)
- Discrimination page (13-method WWV/WWVH voting)
- JSON API endpoints for real-time data
- Server-Sent Events for live updates
"""

from .web_server import WebServer

__all__ = ['WebServer']
