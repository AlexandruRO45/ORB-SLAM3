"""
ORB-SLAM3 Python bindings

This package provides Python bindings for the ORB-SLAM3 visual SLAM system.
"""

__version__ = "1.2.5"
__author__ = "Alex S."
__email__ = "savaalexandru562@gmail.com"

try:
    from .orbslam3 import *  # Import the compiled extension (matches CMake target)
except ImportError as e:
    raise ImportError(
        "Failed to import ORB-SLAM3 core module. "
        "Make sure the package was installed correctly."
    ) from e

# You can add convenience functions here
def get_version():
    """Return the version of the orbslam3 package."""
    return __version__