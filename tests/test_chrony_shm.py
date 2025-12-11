"""
Unit tests for Chrony SHM module.

Tests the shared memory structure packing and unpacking to ensure
compatibility with chronyd's refclock SHM driver.
"""

import pytest
import struct
import time


class TestChronySHMStruct:
    """Test Chrony SHM structure layout matches chronyd expectations."""
    
    def test_shm_size_is_92_bytes(self):
        """Verify SHM struct size matches chronyd's struct shmTime."""
        from time_manager.output.chrony_shm import SHM_SIZE
        assert SHM_SIZE == 92, f"SHM_SIZE should be 92 bytes, got {SHM_SIZE}"
    
    def test_struct_format_produces_92_bytes(self):
        """Verify the struct format string produces correct size."""
        fmt = '@iiqiqiiiiiii8i'
        size = struct.calcsize(fmt)
        assert size == 92, f"Struct format should produce 92 bytes, got {size}"
    
    def test_struct_pack_unpack_roundtrip(self):
        """Verify data survives pack/unpack cycle."""
        fmt = '@iiqiqiiiiiii8i'
        
        # Test values
        mode = 1
        count = 12345
        clock_sec = 1733875200
        clock_usec = 123456
        recv_sec = 1733875200
        recv_usec = 654321
        leap = 0
        precision = -10
        nsamples = 1
        valid = 1
        clock_nsec = 123456789
        recv_nsec = 987654321
        dummy = [0, 0, 0, 0, 0, 0, 0, 0]
        
        # Pack
        data = struct.pack(fmt,
            mode, count, clock_sec, clock_usec,
            recv_sec, recv_usec, leap, precision,
            nsamples, valid, clock_nsec, recv_nsec,
            *dummy
        )
        
        assert len(data) == 92
        
        # Unpack
        unpacked = struct.unpack(fmt, data)
        
        assert unpacked[0] == mode
        assert unpacked[1] == count
        assert unpacked[2] == clock_sec
        assert unpacked[3] == clock_usec
        assert unpacked[4] == recv_sec
        assert unpacked[5] == recv_usec
        assert unpacked[6] == leap
        assert unpacked[7] == precision
        assert unpacked[8] == nsamples
        assert unpacked[9] == valid
        assert unpacked[10] == clock_nsec
        assert unpacked[11] == recv_nsec
    
    def test_field_offsets_match_chronyd(self):
        """Verify field offsets match chronyd's struct shmTime layout."""
        fmt = '@iiqiqiiiiiii8i'
        
        # Pack with identifiable values to check offsets
        data = struct.pack(fmt,
            0x01020304,  # mode (bytes 0-3)
            0x05060708,  # count (bytes 4-7)
            0x1112131415161718,  # clockTimeStampSec (bytes 8-15)
            0x21222324,  # clockTimeStampUSec (bytes 16-19)
            # bytes 20-23 are padding
            0x3132333435363738,  # receiveTimeStampSec (bytes 24-31)
            0x41424344,  # receiveTimeStampUSec (bytes 32-35)
            0x51525354,  # leap (bytes 36-39)
            0x61626364,  # precision (bytes 40-43)
            0x71727374,  # nsamples (bytes 44-47)
            0x01010101,  # valid (bytes 48-51)
            0x11111111,  # clockTimeStampNSec (bytes 52-55)
            0x22222222,  # receiveTimeStampNSec (bytes 56-59)
            0, 0, 0, 0, 0, 0, 0, 0  # dummy (bytes 60-91)
        )
        
        # Verify mode at offset 0
        mode_bytes = struct.unpack('@i', data[0:4])[0]
        assert mode_bytes == 0x01020304
        
        # Verify count at offset 4
        count_bytes = struct.unpack('@i', data[4:8])[0]
        assert count_bytes == 0x05060708
        
        # Verify clockTimeStampSec at offset 8
        clock_sec = struct.unpack('@q', data[8:16])[0]
        assert clock_sec == 0x1112131415161718
        
        # Verify valid at offset 48
        valid_bytes = struct.unpack('@i', data[48:52])[0]
        assert valid_bytes == 0x01010101


class TestChronySHMClass:
    """Test ChronySHM class functionality."""
    
    def test_key_calculation(self):
        """Verify SHM key is calculated correctly."""
        from time_manager.output.chrony_shm import ChronySHM, SHM_KEY_BASE
        
        shm0 = ChronySHM(unit=0)
        assert shm0.key == SHM_KEY_BASE + 0
        
        shm1 = ChronySHM(unit=1)
        assert shm1.key == SHM_KEY_BASE + 1
        
        shm2 = ChronySHM(unit=2)
        assert shm2.key == SHM_KEY_BASE + 2
    
    def test_shm_key_base_is_ntp0(self):
        """Verify SHM key base matches NTP convention (0x4e545030 = 'NTP0')."""
        from time_manager.output.chrony_shm import SHM_KEY_BASE
        
        # 0x4e545030 = 'NTP0' in ASCII
        assert SHM_KEY_BASE == 0x4e545030
        
        # Verify ASCII interpretation
        key_bytes = SHM_KEY_BASE.to_bytes(4, 'big')
        assert key_bytes == b'NTP0'
