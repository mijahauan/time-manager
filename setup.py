#!/usr/bin/env python3
"""Setup configuration for time-manager package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="time-manager",
    version="1.0.0",
    description="Precision HF Time Transfer Daemon for WWV/WWVH/CHU",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Michael James Hauan",
    author_email="ac0g@arrl.net",
    license="MIT",
    
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    python_requires=">=3.9",
    
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "toml>=0.10.0",
    ],
    
    extras_require={
        "chrony": ["sysv_ipc>=1.1.0"],
        "zmq": ["pyzmq>=22.0.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "time-manager=time_manager.main:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: System :: Networking :: Time Synchronization",
    ],
    
    keywords="wwv wwvh chu time synchronization chrony ntp radio",
)
