"""
ORB-SLAM3 Python bindings

This package provides Python bindings for the ORB-SLAM3 visual SLAM system.
"""


from ._version import __version__, __url__, __dependencies__

try:
    # From the file `_core.so`, import the bound C++ classes and enums.
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



# ---- APIs -----
__all__ = [
    "__version__",          # The current version of the compiled ORB-SLAM3 c++ core library file
    "__url__",              # The Github page where the project is located
    "__dependencies__",     # The pip packages must have installed 
    "System",               # The main class for interacting with SLAM
    "IMU",                  # The IMU class for handling inertial measurements
    "Sensor",               # The sensor enum
    "TrackingState",        # The tracking state enum
]