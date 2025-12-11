"""
Pytest configuration and fixtures for time-manager tests.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture
def sample_rate():
    """Standard sample rate for tests."""
    return 20000


@pytest.fixture
def receiver_location():
    """Sample receiver location (Kansas, USA)."""
    return {
        'lat': 38.5,
        'lon': -98.0,
        'grid': 'EM08wl'
    }


@pytest.fixture
def wwv_location():
    """WWV transmitter location."""
    return {
        'lat': 40.6807,
        'lon': -105.0407
    }


@pytest.fixture
def wwvh_location():
    """WWVH transmitter location."""
    return {
        'lat': 21.9872,
        'lon': -159.7636
    }
