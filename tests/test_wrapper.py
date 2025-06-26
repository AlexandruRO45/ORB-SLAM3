# tests/test_wrapper.py

import pytest
import numpy as np
import os
import orbslam3 

# --- Configuration ---
VOCAB_FILE = "tests/configs/ORBvoc.txt"
CONFIGS = {
    "mono": "tests/configs/monocular.yaml",
    "mono_inertial": "tests/configs/mono_inertial.yaml",
    "stereo": "tests/configs/stereo.yaml",
    "stereo_inertial": "tests/configs/stereo_inertial.yaml",
    "rgbd": "tests/configs/rgbd.yaml",
    "rgbd_inertial": "tests/configs/rgbd_inertial.yaml"
}



# --- Helper Functions & Fixtures ---
# Check if path exists for all, if not skip the tests.

def check_files_exist():
    """Checks if the necessary config and vocab files exist."""
    if not os.path.exists(VOCAB_FILE):
        return False
    for config_path in CONFIGS.values():
        if not os.path.exists(config_path):
            return False
    return True

pytestmark = pytest.mark.skipif(
    not check_files_exist(),
    reason=f"Missing required vocab or config files. Please check paths in tests/configs before running test_wrapper.py"
)



# --- Create dummy data for testing ---
# Create dummy images and IMU data for testing purposes.

def create_dummy_image(height=480, width=640, channels=1):
    """Creates a dummy numpy array to simulate an image."""
    if channels == 1:
        return np.random.randint(0, 256, (height, width), dtype=np.uint8)
    else:
        return np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)

def create_dummy_imu_data(timestamp):
    """Creates a list of dummy IMU measurements."""
    imu_points = []
    for i in range(10):
        # IMU timestamps should lead up to the image timestamp
        imu_ts = timestamp - (0.01 * (10 - i))
        imu_points.append(
            orbslam3.IMU.Point(
                acc_x=0.0, acc_y=9.8, acc_z=0.0, # ~1g in Y
                ang_vel_x=0.0, ang_vel_y=0.0, ang_vel_z=0.0,
                timestamp=imu_ts
            )
        )
    return imu_points



# --- Pytest Fixtures ---
# MONO SLAM SYSTEM FIXTURES
@pytest.fixture(scope="function")
def mono_system():
    """Fixture to initialize and shut down a Monocular SLAM system."""
    print("\nSetting up Monocular SLAM system...")
    slam = orbslam3.System(VOCAB_FILE, CONFIGS["mono"], orbslam3.Sensor.MONOCULAR)
    
    assert slam.initialize() is True
    yield slam
    
    print("\nShutting down Monocular SLAM system.")
    slam.shutdown()
    assert slam.is_running() is False

@pytest.fixture(scope="function")
def mono_inertial_system():
    """Fixture to initialize and shut down a Monocular-Inertial SLAM system."""
    print("\nSetting up Monocular-Inertial SLAM system...")
    slam = orbslam3.System(VOCAB_FILE, CONFIGS["mono_inertial"], orbslam3.Sensor.IMU_MONOCULAR)
    
    assert slam.initialize() is True
    yield slam
    
    print("\nShutting down Monocular-Inertial SLAM system.")
    slam.shutdown()
    assert slam.is_running() is False

# STEREO SLAM SYSTEM FIXTURES
@pytest.fixture(scope="function")
def stereo_system():
    """Fixture to initialize and shut down a Monocular SLAM system."""
    print("\nSetting up Monocular SLAM system...")
    slam = orbslam3.System(VOCAB_FILE, CONFIGS["stereo"], orbslam3.Sensor.STEREO)
    
    assert slam.initialize() is True
    yield slam
    
    print("\nShutting down Monocular SLAM system.")
    slam.shutdown()
    assert slam.is_running() is False

@pytest.fixture(scope="function")
def stereo_inertial_system():
    """Fixture to initialize and shut down a Monocular-Inertial SLAM system."""
    print("\nSetting up Monocular-Inertial SLAM system...")
    slam = orbslam3.System(VOCAB_FILE, CONFIGS["stereo_inertial"], orbslam3.Sensor.IMU_STEREO)
    
    assert slam.initialize() is True
    yield slam
    
    print("\nShutting down Monocular-Inertial SLAM system.")
    slam.shutdown()
    assert slam.is_running() is False

# RGB-D SLAM SYSTEM FIXTURES
@pytest.fixture(scope="function")
def stereo_system():
    """Fixture to initialize and shut down a Monocular SLAM system."""
    print("\nSetting up Monocular SLAM system...")
    slam = orbslam3.System(VOCAB_FILE, CONFIGS["rgbd"], orbslam3.Sensor.RGBD)
    
    assert slam.initialize() is True
    yield slam
    
    print("\nShutting down Monocular SLAM system.")
    slam.shutdown()
    assert slam.is_running() is False

@pytest.fixture(scope="function")
def stereo_inertial_system():
    """Fixture to initialize and shut down a Monocular-Inertial SLAM system."""
    print("\nSetting up Monocular-Inertial SLAM system...")
    slam = orbslam3.System(VOCAB_FILE, CONFIGS["rgbd_inertial"], orbslam3.Sensor.IMU_RGBD)
    
    assert slam.initialize() is True
    yield slam
    
    print("\nShutting down Monocular-Inertial SLAM system.")
    slam.shutdown()
    assert slam.is_running() is False



class TestSystemLifecycle:
    """Tests the creation, initialization, and shutdown of the SLAM object."""

    def test_system_creation(self):
        """Verify that the system object can be instantiated without error."""
        slam = orbslam3.System(VOCAB_FILE, CONFIGS["mono"], orbslam3.Sensor.MONOCULAR)
        
        assert slam is not None, "System object should be created."
        assert not slam.is_running(), "System should not be running before initialization."

    def test_initialization_and_shutdown(self):
        """Test the initialize() and shutdown() calls."""
        slam = orbslam3.System(VOCAB_FILE, CONFIGS["mono"], orbslam3.Sensor.MONOCULAR)
        
        assert slam.initialize(), "Initialization should return True."
        assert slam.is_running(), "System should be running after initialization."
        
        slam.shutdown()
        assert not slam.is_running(), "System should not be running after shutdown."



class TestDataRetrieval:
    """Tests all 'getter' methods that retrieve data from the SLAM system."""

    def test_get_initial_state(self, initialized_mono_system):
        """Test the initial values returned by getter methods."""
        slam = initialized_mono_system
        
        assert slam.get_tracking_state() in [orbslam3.TrackingState.SYSTEM_NOT_READY, orbslam3.TrackingState.NO_IMAGES_YET]
        assert isinstance(slam.is_lost(), bool)
        assert slam.get_pose().shape == (4, 4)
        assert np.all(slam.get_pose() == np.eye(4)), "Initial pose should be an identity matrix."
        assert slam.get_trajectory() == [], "Initial trajectory should be empty."
        assert slam.get_reset_count() == 0

    def test_get_state_after_frame(self, initialized_mono_system):
        """Test getters after processing one frame."""
        slam = initialized_mono_system
        dummy_image = create_dummy_image()
        timestamp = 1000.0
        slam.process_image_mono(dummy_image, timestamp)
        
        assert isinstance(slam.get_tracking_state(), orbslam3.TrackingState)
        assert isinstance(slam.get_pose(), np.ndarray)
        assert slam.get_pose().shape == (4, 4)
        assert isinstance(slam.get_trajectory(), list)

        if slam.get_trajectory():
            assert slam.get_trajectory()[0].shape == (4, 4)
    
    def test_get_2d_occ_map(self, initialized_mono_system):
        """Tests that get_2d_occmap returns a valid 2D numpy array."""
        slam = initialized_mono_system
        dummy_image = create_dummy_image()
        timestamp = 1000.0
        slam.process_image_mono(dummy_image, timestamp)        
        occ_map = slam.get_2d_occmap()
        
        assert isinstance(occ_map, np.ndarray), "Occupancy map should be a NumPy array."
        assert occ_map.ndim == 2, "Occupancy map should be 2-dimensional."
        assert occ_map.dtype == np.int16, "Occupancy map data type should be int16."



class TestSystemControl:
    """Tests methods that control the system's behavior, like reset."""

    def test_reset(self, initialized_mono_system):
        """Test the map reset functionality."""
        slam = initialized_mono_system
        dummy_image = create_dummy_image()
        timestamp = 1001.0
        
        assert slam.get_reset_count() == 0
        assert not slam.was_map_reset()
        
        slam.process_image_mono(dummy_image, timestamp)
        slam.reset()
        
        assert slam.get_reset_count() >= 1, "Reset counter should increment."
        assert slam.was_map_reset(), "was_map_reset flag should be true immediately after reset."
        assert not slam.was_map_reset(), "was_map_reset flag should be consumed after checking."
    
    def test_set_use_viewer(self):
        """Tests the setUseViewer method. This is a simple check."""

        slam = orbslam3.system(VOCAB_FILE, CONFIGS["mono"], orbslam3.Sensor.MONOCULAR)
        slam.set_use_viewer(True)
        assert True 



@pytest.mark.parametrize("sensor_details", [
    {"name": "mono", "sensor_enum": orbslam3.Sensor.MONOCULAR, "config": CONFIGS["mono"]},
    {"name": "stereo", "sensor_enum": orbslam3.Sensor.STEREO, "config": CONFIGS["stereo"]},
    {"name": "rgbd", "sensor_enum": orbslam3.Sensor.RGBD, "config": CONFIGS["rgbd"]},
    {"name": "mono_inertial", "sensor_enum": orbslam3.Sensor.IMU_MONOCULAR, "config": CONFIGS["mono_inertial"]},
    # TODO: Uncomment when stereo_inertial and rgbd_inertial are implemented
    # {"name": "stereo_inertial", "sensor_enum": orbslam3.Sensor.IMU_STEREO, "config": CONFIGS["stereo_inertial"]},
    # {"name": "rgbd_inertial", "sensor_enum": orbslam3.Sensor.IMU_RGBD, "config": CONFIGS["rgbd_inertial"]}
])
class TestProcessingModes:
    """Contains smoke tests for each sensor processing mode, using helper functions."""

    def test_process_frame(self, sensor_details):
        """A generic test to ensure each processing mode can be called without crashing."""
        slam = orbslam3.System(VOCAB_FILE, sensor_details["config"], sensor_details["sensor_enum"])
        slam.initialize()
        
        timestamp = 2000.0
        image_left = create_dummy_image()

        try:
            if sensor_details["name"] == "mono":
                slam.process_image_mono(image_left, timestamp)
            elif sensor_details["name"] == "stereo":
                image_right = create_dummy_image() 
                slam.process_image_stereo(image_left, image_right, timestamp)
            elif sensor_details["name"] == "rgbd":
                depth_img = create_dummy_image().astype(np.uint16) * 10 
                slam.process_image_rgbd(image_left, depth_img, timestamp)
            elif sensor_details["name"] == "mono_inertial":
                imu_data = create_dummy_imu_data(timestamp)
                slam.process_image_mono_inertial(image_left, timestamp, imu_data)
            # TODO: Uncomment when stereo_inertial and rgbd_inertial are implemented
            # elif sensor_details["name"] == "stereo_inertial":
            #     image_right = create_dummy_image()
            #     imu_data = create_dummy_imu_data(timestamp)
            #     slam.process_image_stereo_inertial(image_left, image_right, timestamp, imu_data)
            # elif sensor_details["name"] == "rgbd_inertial":
            #     depth_img = create_dummy_image().astype(np.uint16) * 10
            #     imu_data = create_dummy_imu_data(timestamp)
            #     slam.process_image_rgbd_inertial(image_left, depth_img, timestamp, imu_data)
            
            pose = slam.get_pose()
            assert pose.shape == (4, 4)

        finally:
            slam.shutdown()

class TestIMUHelpers:
    """Tests the IMU.Point data structure exposed from C++."""

    def test_imu_point_creation_and_access(self):
        """Verify IMU.Point can be created and its members accessed."""
        ts = 12345.6789
        p = orbslam3.IMU.Point(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, ts)
        assert p.ax == 1.0
        assert p.wy == 5.0
        assert p.t == ts
    
    
    
    
    
    
    
    
    
    
    


# # --- Placeholder tests for other sensor types ---
# # You can implement these in the same way once you have the config files.
# @pytest.mark.skip(reason="Stereo settings file path needs to be configured.")
# def test_process_stereo_frame():
#     # 1. Create a stereo fixture similar to mono_system
#     # 2. Create two dummy images (left and right)
#     # 3. Call process_image_stereo() and assert results
#     pass

# @pytest.mark.skip(reason="RGB-D settings file path needs to be configured.")
# def test_process_rgbd_frame():
#     # 1. Create an RGB-D fixture
#     # 2. Create a dummy color image and a dummy depth image (e.g., dtype=np.uint16)
#     # 3. Call process_image_rgbd() and assert results
#     pass