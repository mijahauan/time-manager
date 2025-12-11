"""
Unit tests for Transmission Time Solver module.

Tests ionospheric delay factors, spherical Earth geometry,
and propagation mode calculations.
"""

import pytest
import math


class TestIonosphericDelayFactors:
    """Test ionospheric delay factor calculations."""
    
    def test_delay_factor_follows_inverse_square_law(self):
        """Verify delay factors follow 1/f² physics."""
        from time_manager.timing.transmission_time_solver import (
            IONO_DELAY_FACTOR, _iono_delay_factor
        )
        
        # 2.5 MHz should have (10/2.5)² = 16× more delay than 10 MHz
        assert abs(IONO_DELAY_FACTOR[2.5] - 16.0) < 0.01
        
        # 5 MHz should have (10/5)² = 4× more delay
        assert abs(IONO_DELAY_FACTOR[5.0] - 4.0) < 0.01
        
        # 10 MHz is reference (factor = 1.0)
        assert abs(IONO_DELAY_FACTOR[10.0] - 1.0) < 0.01
        
        # 20 MHz should have (10/20)² = 0.25× delay
        assert abs(IONO_DELAY_FACTOR[20.0] - 0.25) < 0.01
    
    def test_delay_factor_function(self):
        """Test the delay factor calculation function."""
        from time_manager.timing.transmission_time_solver import _iono_delay_factor
        
        # Reference frequency
        assert _iono_delay_factor(10.0) == 1.0
        
        # Half frequency = 4× delay
        assert _iono_delay_factor(5.0) == 4.0
        
        # Double frequency = 0.25× delay
        assert _iono_delay_factor(20.0) == 0.25
        
        # Custom reference
        assert _iono_delay_factor(5.0, ref_freq=5.0) == 1.0


class TestSphericalEarthGeometry:
    """Test spherical Earth geometry calculations."""
    
    def test_short_path_uses_flat_earth(self):
        """Verify short paths (< 500 km) use flat-Earth approximation."""
        from time_manager.timing.transmission_time_solver import TransmissionTimeSolver
        
        solver = TransmissionTimeSolver(
            receiver_lat=40.0,
            receiver_lon=-100.0,
            enable_dynamic_ionosphere=False
        )
        
        # 400 km path should use flat-Earth
        path, elev = solver._calculate_hop_path(400, 300, 1)
        
        # Flat-Earth calculation: slant = sqrt(200² + 300²) = 360.56 km
        # Path = 2 × slant = 721.1 km
        expected_path = 2 * math.sqrt(200**2 + 300**2)
        assert abs(path - expected_path) < 0.1
    
    def test_long_path_uses_spherical_earth(self):
        """Verify long paths (>= 500 km) use spherical geometry."""
        from time_manager.timing.transmission_time_solver import TransmissionTimeSolver
        
        solver = TransmissionTimeSolver(
            receiver_lat=40.0,
            receiver_lon=-100.0,
            enable_dynamic_ionosphere=False
        )
        
        # 4000 km path should use spherical geometry
        path_spherical, elev_spherical = solver._calculate_hop_path(4000, 300, 2)
        
        # Compare with what flat-Earth would give
        # Flat: hop_dist = 2000, half_hop = 1000
        # slant = sqrt(1000² + 300²) = 1044.03 km
        # path = 2 × 1044.03 × 2 = 4176.1 km
        flat_path = 2 * math.sqrt(1000**2 + 300**2) * 2
        
        # Spherical should give a LONGER path due to Earth curvature
        assert path_spherical > flat_path
        
        # The difference should be noticeable but not huge (~1-3%)
        diff_percent = (path_spherical - flat_path) / flat_path * 100
        assert 0.5 < diff_percent < 5.0
    
    def test_ground_wave_returns_ground_distance(self):
        """Verify ground wave mode returns ground distance."""
        from time_manager.timing.transmission_time_solver import TransmissionTimeSolver
        
        solver = TransmissionTimeSolver(
            receiver_lat=40.0,
            receiver_lon=-100.0,
            enable_dynamic_ionosphere=False
        )
        
        path, elev = solver._calculate_hop_path(150, 0, 0)
        
        assert path == 150
        assert elev == 0.0
    
    def test_elevation_angle_is_reasonable(self):
        """Verify elevation angles are physically reasonable."""
        from time_manager.timing.transmission_time_solver import TransmissionTimeSolver
        
        solver = TransmissionTimeSolver(
            receiver_lat=40.0,
            receiver_lon=-100.0,
            enable_dynamic_ionosphere=False
        )
        
        # Short hop with high layer = steep angle
        _, elev_steep = solver._calculate_hop_path(200, 300, 1)
        assert 50 < elev_steep < 80  # Should be steep
        
        # Long single-hop = shallower angle
        _, elev_shallow = solver._calculate_hop_path(2000, 300, 1)
        assert 10 < elev_shallow < 25  # Should be shallower
        
        # Multi-hop long path: each hop is shorter, so steeper per-hop angle
        _, elev_multihop = solver._calculate_hop_path(4000, 300, 2)
        # 2 hops over 4000 km = 2000 km per hop, similar to single 2000 km hop
        assert elev_multihop > 0
        
        # All elevations should be non-negative
        assert elev_steep >= 0
        assert elev_shallow >= 0
        assert elev_multihop >= 0


class TestPropagationModes:
    """Test propagation mode enumeration and constraints."""
    
    def test_propagation_modes_exist(self):
        """Verify all expected propagation modes are defined."""
        from time_manager.timing.transmission_time_solver import PropagationMode
        
        assert PropagationMode.GROUND_WAVE.value == "GW"
        assert PropagationMode.ONE_HOP_E.value == "1E"
        assert PropagationMode.ONE_HOP_F.value == "1F"
        assert PropagationMode.TWO_HOP_F.value == "2F"
        assert PropagationMode.THREE_HOP_F.value == "3F"
        assert PropagationMode.MIXED_EF.value == "EF"
        assert PropagationMode.UNKNOWN.value == "UNK"


class TestStationDistances:
    """Test station distance calculations."""
    
    def test_station_locations_imported(self):
        """Verify station locations are properly imported."""
        from time_manager.timing.transmission_time_solver import STATIONS
        
        assert 'WWV' in STATIONS
        assert 'WWVH' in STATIONS
        assert 'CHU' in STATIONS
        
        # WWV is in Colorado
        assert 39 < STATIONS['WWV']['lat'] < 42
        assert -106 < STATIONS['WWV']['lon'] < -104
        
        # WWVH is in Hawaii
        assert 21 < STATIONS['WWVH']['lat'] < 23
        assert -160 < STATIONS['WWVH']['lon'] < -159
        
        # CHU is in Ottawa
        assert 45 < STATIONS['CHU']['lat'] < 46
        assert -76 < STATIONS['CHU']['lon'] < -75
