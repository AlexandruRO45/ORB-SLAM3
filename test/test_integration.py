import pytest
import numpy as np
from pathlib import Path
import cv2
import orbslam3 

# Define paths relative to the project root
PROJECT_ROOT = Path(__file__).parent.parent
VOCAB_FILE = PROJECT_ROOT / "third_party" / "ORB_SLAM3_engine" / "Vocabulary" / "ORBvoc.txt"
# NOTE: Using a Monocular settings file to match the MONOCULAR sensor type
SETTINGS_FILE = PROJECT_ROOT / "third_party" / "ORB_SLAM3_engine" / "Examples" / "Monocular" / "TUM1.yaml"
TEST_DATASET_PATH = PROJECT_ROOT / "test" / "data" / "TUM1_xyz"

@pytest.mark.integration
def test_monocular_run():
    """
    Integration test that simulates a full run on a small dataset.
    This test ensures the system can initialize, process a sequence of frames,
    and shut down without errors.
    """
    # --- 1. Check for required files ---
    assert VOCAB_FILE.exists(), f"Vocabulary file not found at {VOCAB_FILE}"
    assert SETTINGS_FILE.exists(), f"Settings file not found at {SETTINGS_FILE}"
    assert TEST_DATASET_PATH.exists(), f"Test dataset not found at {TEST_DATASET_PATH}"

    image_files = sorted(list((TEST_DATASET_PATH / "rgb").glob('*.png')))
    assert len(image_files) > 0, "No images found in test dataset"

    # --- 2. Initialize the SLAM System ---
    # Based on your script's API: orbslam3.System
    # Note: Your script had a mismatch (MONOCULAR sensor with RGBD settings). 
    # This test uses a consistent MONOCULAR setup.
    slam = orbslam3.System(str(VOCAB_FILE), str(SETTINGS_FILE), orbslam3.Sensor.MONOCULAR)
    
    # Disable the viewer
    slam.set_use_viewer(False)
    
    # This seems to be part of your API from the script
    slam.initialize()

    # --- 3. Process all frames in the dataset ---
    initial_pose = slam.process_image_mono(cv2.imread(str(image_files[0]), -1), 0.0)

    for i, img_path in enumerate(image_files[1:]):
        timestamp = float(img_path.stem) # Extract timestamp from filename
        img = cv2.imread(str(img_path), -1)
        assert img is not None, f"Failed to load image: {img_path}"
        
        # Process the image
        current_pose = slam.process_image_mono(img, timestamp)
    
    # --- 4. Assertions ---
    # The main test is that the loop above completes without crashing.
    # We can also add a basic check on the final pose.
    assert current_pose is not None
    assert isinstance(current_pose, np.ndarray)
    assert current_pose.shape == (4, 4)
    
    # Check that the pose is not the identity matrix, which would mean tracking failed
    assert not np.allclose(current_pose, np.identity(4))
    
    # --- 5. Shutdown ---
    slam.shutdown()
    print("\nSLAM system shutdown successfully.")