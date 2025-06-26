"""
ORB-SLAM3 Python bindings

This package provides Python bindings for the ORB-SLAM3 visual SLAM system.
"""

# 1. Import the version for user access
from ._version import __version__

# 2. Try to import the compiled C++ core and its contents.
try:
    # From the file `_core.so`, import the bound C++ classes and enums.
    # We alias the lowercase "system" class to a more Pythonic "System"
    from ._core import system as System
    from ._core import IMU, Sensor, TrackingState

except ImportError as e:
    # This provides a much better error message if the C++ part failed.
    # Include the original error for more detailed debugging.
    raise ImportError(
        "Failed to import the compiled ORB-SLAM3 C++ core (_core.so).\n"
        "Please make sure the package was installed correctly after a full compilation.\n"
        f"Original error: {e}"
    ) from e

# 3. Define the public API of the package.
__all__ = [
    "__version__",
    "System",         # The main class for interacting with SLAM
    "IMU",            # The IMU class for handling inertial measurements
    "Sensor",         # The sensor enum
    "TrackingState",  # The tracking state enum
]