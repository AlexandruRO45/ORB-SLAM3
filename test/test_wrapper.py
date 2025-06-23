# tests/test_wrapper.py
import pytest
import orbslam3 # This will import from the installed or built package
import numpy as np
import os

# To run these test:
# 1. pip install pytest
# 2. Download ORB-SLAM3 vocabulary and a settings file.
# 3. export ORBSLAM_DATA_PATH=/path/to/your/data
# 4. Run `pytest` in the terminal from your project's root directory.


# Mark that this test requires data files, skip if not found
VOCAB_FILE = os.path.join(os.getenv("ORBSLAM_DATA_PATH", "."), "ORBvoc.txt")
SETTINGS_FILE = os.path.join(os.getenv("ORBSLAM_DATA_PATH", "."), "TUM1.yaml")

# A decorator to skip tests if essential data files are missing
requires_data = pytest.mark.skipif(
    not all(os.path.exists(f) for f in [VOCAB_FILE, SETTINGS_FILE]),
    reason="Data files (ORBvoc.txt, TUM1.yaml) not found. Set ORBSLAM_DATA_PATH env var."
)

@requires_data
def test_slam_initialization():
    """
    Tests if the SLAM system can be initialized without errors.
    """
    try:
        slam = orbslam3.ORBSLAM3(VOCAB_FILE, SETTINGS_FILE)
        assert slam is not None, "SLAM object should not be None"
    except Exception as e:
        pytest.fail(f"SLAM initialization failed with an exception: {e}")

@requires_data
def test_process_frame_dummy_data():
    """
    Tests the frame processing method with dummy data.
    This is a placeholder - a real test would use actual image data.
    """
    slam = orbslam3.ORBSLAM3(VOCAB_FILE, SETTINGS_FILE)
    
    # Create a dummy grayscale image (e.g., 640x480)
    dummy_image = np.zeros((480, 640), dtype=np.uint8)
    timestamp = 0.0

    try:
        # Assuming your wrapper has a method like `process_image_mono`
        pose = slam.process_image_mono(dummy_image, timestamp)
        assert isinstance(pose, np.ndarray), "Pose should be a numpy array"
        assert pose.shape == (4, 4), "Pose matrix should be 4x4"
    except Exception as e:
        pytest.fail(f"Processing a dummy frame failed with an exception: {e}")

