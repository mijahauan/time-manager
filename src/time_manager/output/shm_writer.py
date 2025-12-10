"""
Shared Memory Writer for Time Manager

Writes TimingResult to /dev/shm/grape_timing for consumption by
grape-recorder and other applications.

The shared memory file is updated atomically (write to temp, rename)
to prevent partial reads.

Usage:
    writer = SHMWriter('/dev/shm/grape_timing')
    writer.write(timing_result)
"""

import json
import os
import tempfile
import logging
from pathlib import Path
from typing import Optional

from ..interfaces.timing_result import TimingResult

logger = logging.getLogger(__name__)


class SHMWriter:
    """
    Writes TimingResult to shared memory file.
    
    The file is JSON-formatted for easy debugging and consumption.
    Updates are atomic (write to temp file, then rename).
    """
    
    DEFAULT_PATH = "/dev/shm/grape_timing"
    
    def __init__(self, shm_path: Optional[str] = None):
        """
        Initialize SHM writer.
        
        Args:
            shm_path: Path to shared memory file (default: /dev/shm/grape_timing)
        """
        self.shm_path = Path(shm_path or self.DEFAULT_PATH)
        self.write_count = 0
        
        # Ensure parent directory exists (it should for /dev/shm)
        self.shm_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"SHMWriter initialized: {self.shm_path}")
    
    def write(self, result: TimingResult) -> bool:
        """
        Write timing result to shared memory.
        
        Uses atomic write (temp file + rename) to prevent partial reads.
        
        Args:
            result: TimingResult to write
            
        Returns:
            True if successful, False on error
        """
        try:
            # Serialize to JSON
            json_data = result.to_json()
            
            # Write to temp file in same directory (required for atomic rename)
            fd, temp_path = tempfile.mkstemp(
                dir=self.shm_path.parent,
                prefix='.grape_timing_',
                suffix='.tmp'
            )
            
            try:
                with os.fdopen(fd, 'w') as f:
                    f.write(json_data)
                
                # Atomic rename
                os.rename(temp_path, self.shm_path)
                
                self.write_count += 1
                
                if self.write_count % 60 == 0:  # Log every 60 writes (~1 minute)
                    logger.debug(
                        f"SHM write #{self.write_count}: "
                        f"d_clock={result.d_clock_ms:+.2f}ms, "
                        f"status={result.clock_status.value}"
                    )
                
                return True
                
            except Exception as e:
                # Clean up temp file on error
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise
                
        except Exception as e:
            logger.error(f"Failed to write SHM: {e}")
            return False
    
    def read(self) -> Optional[TimingResult]:
        """
        Read current timing result from shared memory.
        
        Useful for self-verification or status queries.
        
        Returns:
            TimingResult or None if file doesn't exist or is invalid
        """
        try:
            if not self.shm_path.exists():
                return None
            
            with open(self.shm_path, 'r') as f:
                json_data = f.read()
            
            return TimingResult.from_json(json_data)
            
        except Exception as e:
            logger.warning(f"Failed to read SHM: {e}")
            return None
    
    def clear(self):
        """Remove shared memory file."""
        try:
            if self.shm_path.exists():
                self.shm_path.unlink()
                logger.info(f"Cleared SHM: {self.shm_path}")
        except Exception as e:
            logger.warning(f"Failed to clear SHM: {e}")


class SHMReader:
    """
    Reads TimingResult from shared memory.
    
    This is the client-side reader used by grape-recorder and other
    applications to consume timing data from time-manager.
    
    Usage:
        reader = SHMReader('/dev/shm/grape_timing')
        result = reader.read()
        if result and result.clock_status == ClockStatus.LOCKED:
            d_clock = result.d_clock_ms
    """
    
    DEFAULT_PATH = "/dev/shm/grape_timing"
    
    def __init__(self, shm_path: Optional[str] = None):
        """
        Initialize SHM reader.
        
        Args:
            shm_path: Path to shared memory file
        """
        self.shm_path = Path(shm_path or self.DEFAULT_PATH)
        self._last_timestamp = 0.0
        self._read_count = 0
    
    def read(self) -> Optional[TimingResult]:
        """
        Read current timing result.
        
        Returns:
            TimingResult or None if unavailable
        """
        try:
            if not self.shm_path.exists():
                return None
            
            with open(self.shm_path, 'r') as f:
                json_data = f.read()
            
            result = TimingResult.from_json(json_data)
            self._last_timestamp = result.timestamp
            self._read_count += 1
            
            return result
            
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in SHM: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to read SHM: {e}")
            return None
    
    def get_d_clock(self) -> Optional[float]:
        """
        Convenience method to get just the D_clock value.
        
        Returns:
            D_clock in milliseconds, or None if unavailable
        """
        result = self.read()
        if result and result.clock_status in (
            ClockStatus.LOCKED, ClockStatus.HOLDOVER
        ):
            return result.d_clock_ms
        return None
    
    def get_channel_station(self, channel_name: str) -> Optional[str]:
        """
        Get the identified station for a channel.
        
        Args:
            channel_name: Channel name (e.g., "WWV 10 MHz")
            
        Returns:
            Station name ("WWV", "WWVH", "CHU") or None
        """
        result = self.read()
        if result:
            # Try exact match first
            if channel_name in result.channels:
                return result.channels[channel_name].station
            
            # Try normalized key (underscores)
            key = channel_name.replace(' ', '_')
            if key in result.channels:
                return result.channels[key].station
        
        return None
    
    def is_locked(self) -> bool:
        """Check if time-manager has achieved lock."""
        result = self.read()
        return result is not None and result.clock_status == ClockStatus.LOCKED
    
    @property
    def available(self) -> bool:
        """Check if shared memory file exists."""
        return self.shm_path.exists()
