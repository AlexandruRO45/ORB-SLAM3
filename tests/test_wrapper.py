# tests/test_wrapper.py

import pytest
import numpy as np
import os
import orbslam3 
import cv2

# --- Configuration ---
VOCAB_FILE = "tests/configs/ORBvoc.txt"
CONFIGS = {
    "mono": "tests/configs/mono.yaml",
    "mono_inertial": "tests/configs/mono_inertial.yaml",
    "stereo": "tests/configs/stereo.yaml",
    "stereo_inertial": "tests/configs/stereo_inertial.yaml",
    "rgbd": "tests/configs/rgbd.yaml",
    "rgbd_inertial": "tests/configs/rgbd_inertial.yaml"
}

# --- Helper Functions & Prerequisite Checks ---
def check_files_exist():
    """Checks if the necessary config and vocab files exist."""
    if not os.path.exists(VOCAB_FILE):
        return False
    # TODO: Replace once all configs modes are in place
    # for config_path in CONFIGS.values():
    #     if not os.path.exists(config_path):
    #         return False
    active_configs = ["mono", "mono_inertial", "stereo", "rgbd"]
    for key in active_configs:
        if not os.path.exists(CONFIGS[key]):
            return False
    return True


# --- Skip tests if files are missing ---
pytestmark = pytest.mark.skipif(
    not check_files_exist(),
    reason="Missing required vocab or config files. Please check paths in tests/configs."
)

# --- Create structured dummy data for testing ---
def create_dummy_image(height=480, width=640):
    """
    Creates a more realistic dummy image with simple shapes.
    This is more stable for feature extractors than pure random noise.
    """
    # Create a black image
    img = np.zeros((height, width), dtype=np.uint8)
    
    # Draw some white shapes for the feature extractor to find
    cv2.rectangle(img, (50, 50), (150, 150), 255, -1)  # A solid square
    cv2.line(img, (100, 200), (400, 250), 255, 5)      # A thick line
    cv2.circle(img, (500, 350), 50, 255, -1)           # A solid circle
    
    return img

def create_dummy_depth_image(height=480, width=640):
    """
    Creates a more realistic dummy depth image with structured depth regions.
    Mimics a scene with flat surfaces at different depths.
    """
    depth_img = np.zeros((height, width), dtype=np.uint16)

    # Simulate a floor plane
    cv2.rectangle(depth_img, (0, int(height*0.6)), (width, height), 1000, -1)
    cv2.rectangle(depth_img, (50, 50), (200, 300), 500, -1)   # Closer object
    cv2.rectangle(depth_img, (300, 100), (550, 350), 800, -1) # Further object

    for y in range(height):
        depth_img[y] += int(100 * (y / height))  # linear depth gradient
    
    return depth_img

def create_dummy_imu_data(timestamp, num_samples=10):
    """Creates a list of dummy IMU measurements."""
    imu_points = []
    for i in range(num_samples):
        imu_ts = timestamp - (0.01 * (num_samples - i))
        imu_points.append(
            orbslam3.IMU.Point(
                acc_x=0.0, acc_y=9.8, acc_z=0.0,
                ang_vel_x=0.0, ang_vel_y=0.0, ang_vel_z=0.0,
                timestamp=imu_ts
            )
        )
    return imu_points

# --- Consolidated Fixtures ---
@pytest.fixture(scope="function", params=[
    pytest.param(("mono", orbslam3.Sensor.MONOCULAR), id="MONOCULAR"),
    pytest.param(("mono_inertial", orbslam3.Sensor.IMU_MONOCULAR), id="MONO_INERTIAL"),
    pytest.param(("stereo", orbslam3.Sensor.STEREO), id="STEREO"),
    pytest.param(("rgbd", orbslam3.Sensor.RGBD), id="RGBD"),
])
def system_for_all_modes(request):
    """
    A single, parameterized fixture that initializes a SLAM system.
    It relies on the C++ destructor for safe, automatic shutdown.
    """
    config_key, sensor_type = request.param
    system_name = request.node.callspec.id

    slam = orbslam3.System(VOCAB_FILE, CONFIGS[config_key], sensor_type)
    assert slam.initialize() is True, f"Failed to initialize {system_name} system."
    
    yield slam, config_key

# --- Test Classes ---
class TestSystemLifecycle:
    """Tests the creation, initialization, and shutdown of the SLAM object."""

    def test_system_creation(self):
        """Verify that the system object can be instantiated without error."""
        slam = orbslam3.System(VOCAB_FILE, CONFIGS["mono"], orbslam3.Sensor.MONOCULAR)
        assert slam is not None
        assert not slam.is_running()

    def test_system_is_initialized_and_running(self, system_for_all_modes):
        """
        Receives an initialized system from the fixture and verifies its state.
        """
        slam, _ = system_for_all_modes
        assert slam.is_running() is True
        assert slam.get_tracking_state() in [
            orbslam3.TrackingState.SYSTEM_NOT_READY,
            orbslam3.TrackingState.NO_IMAGES_YET
        ]

class TestDataRetrieval:
    """Tests all 'getter' methods that retrieve data from the SLAM system for all modes."""

    def test_get_initial_state(self, system_for_all_modes):
        """Test the initial values returned by getter methods for all modes."""
        slam, _ = system_for_all_modes
        assert slam.get_tracking_state() in [orbslam3.TrackingState.SYSTEM_NOT_READY, orbslam3.TrackingState.NO_IMAGES_YET]
        assert isinstance(slam.is_lost(), bool)
        assert np.all(slam.get_pose() == np.eye(4))
        assert slam.get_trajectory() == []
        assert slam.get_reset_count() == 0

    def test_get_state_after_frame(self, system_for_all_modes):
        """Test getters after processing one frame, using the correct process for each mode."""
        slam, system_type = system_for_all_modes
        timestamp = 1000.0

        if system_type == "mono":
            slam.process_image_mono(create_dummy_image(), timestamp)
        elif system_type == "stereo":
            slam.process_image_stereo(create_dummy_image(), create_dummy_image(), timestamp)
        elif system_type == "rgbd":
            slam.process_image_rgbd(create_dummy_image(), create_dummy_depth_image(), timestamp)
        elif system_type == "mono_inertial":
            slam.process_image_mono_inertial(create_dummy_image(), timestamp, create_dummy_imu_data(timestamp))
        else:
            pytest.fail(f"Test logic for advancing state of '{system_type}' is not implemented.")
            
        assert isinstance(slam.get_tracking_state(), orbslam3.TrackingState)

class TestSystemControl:
    """Tests methods that control the system's behavior, like reset."""

    def test_reset(self, system_for_all_modes):
        """Test the map reset functionality on all sensor modes."""
        slam, system_type = system_for_all_modes
        timestamp = 1001.0

        if system_type == "mono":
            slam.process_image_mono(create_dummy_image(), timestamp)
        elif system_type == "stereo":
            slam.process_image_stereo(create_dummy_image(), create_dummy_image(), timestamp)
        elif system_type == "rgbd":
            slam.process_image_rgbd(create_dummy_image(), create_dummy_depth_image(), timestamp)
        elif system_type == "mono_inertial":
            slam.process_image_mono_inertial(create_dummy_image(), timestamp, create_dummy_imu_data(timestamp))
        else:
            pytest.fail(f"Test logic for advancing state of '{system_type}' is not implemented.")

        initial_reset_count = slam.get_reset_count()
        slam.reset()
        assert slam.get_reset_count() > initial_reset_count
        assert slam.was_map_reset() is True

    def test_set_use_viewer(self):
        """Tests the setUseViewer method can be called without error."""
        slam = orbslam3.System(VOCAB_FILE, CONFIGS["mono"], orbslam3.Sensor.MONOCULAR)
        slam.set_use_viewer(True)
        assert True

class TestIMUHelpers:
    """Tests the IMU.Point data structure exposed from C++."""

    def test_imu_point_creation_and_access(self):
        """Verify IMU.Point can be created and its members accessed."""
        ts = 12345.6789
        p = orbslam3.IMU.Point(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, ts)
        assert p.ax == 1.0
        assert p.wy == 5.0
        assert p.t == ts
