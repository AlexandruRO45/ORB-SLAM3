# tests/test_wrapper.py

import pytest
import numpy as np
import os
import orbslam3 

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
    
def create_dummy_depth_image(height=480, width=640, channels=1):
    """Creates a dummy depth image (16-bit)."""
    if channels == 1:
        return (np.random.rand(height, width) * 1000).astype(np.uint16)
    else:
        return (np.random.rand(height, width, channels) * 1000).astype(np.uint16)

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


# TODO: Uncomment when stereo_inertial and rgbd_inertial are implemented
@pytest.fixture(scope="function", params=[
    pytest.param(("mono", orbslam3.Sensor.MONOCULAR), id="MONOCULAR"),
    pytest.param(("mono_inertial", orbslam3.Sensor.IMU_MONOCULAR), id="MONO_INERTIAL"),
    pytest.param(("stereo", orbslam3.Sensor.STEREO), id="STEREO"),
    # pytest.param(("stereo_inertial", orbslam3.Sensor.IMU_STEREO), id="STEREO_INERTIAL"),
    pytest.param(("rgbd", orbslam3.Sensor.RGBD), id="RGBD"),
    # pytest.param(("rgbd_inertial", orbslam3.Sensor.IMU_RGBD), id="RGBD_INERTIAL"),
])
def slam_system_and_type(request):
    """
    A single, parameterized fixture that initializes and shuts down any
    type of SLAM system based on the provided parameters.
    """
    config_key, sensor_type = request.param
    system_name = request.node.callspec.id  # Gets the 'id' from pytest.param

    print(f"\nSetting up {system_name} SLAM system...")
    slam = orbslam3.System(VOCAB_FILE, CONFIGS[config_key], sensor_type)
    
    assert slam.initialize() is True, f"Failed to initialize {system_name} system."
    yield slam  
    
    print(f"\nShutting down {system_name} SLAM system.")
    slam.shutdown()
    assert slam.is_running() is False, f"Failed to shut down {system_name} system."



class TestAllSystemModes:
    """
    Tests all sensor modes by adapting the test logic based on the
    specific system type provided by the parameterized fixture.
    """
    
    def test_process_first_frame_for_each_mode(self, slam_system_and_type):
        """
        This single test method is executed for each system type. It correctly
        calls the appropriate 'process_image_*' method with the right data.
        """
        # 1. Arrange: Unpack the tuple from the fixture
        slam_system, system_type = slam_system_and_type
        timestamp = 1000.0
        
        print(f"Running test on a '{system_type}' system.")
        
        # Assert initial state (from fixture)
        assert slam_system.is_running() is True
        assert slam_system.get_tracking_state() in [
            orbslam3.TrackingState.SYSTEM_NOT_READY,
            orbslam3.TrackingState.NO_IMAGES_YET
        ]

        # 2. Act: Call the correct processing method based on the system type
        if system_type == "mono":
            slam_system.process_image_mono(create_dummy_image(), timestamp)
        
        elif system_type == "stereo":
            slam_system.process_image_stereo(create_dummy_image(), create_dummy_image(), timestamp)

        elif system_type == "rgbd":
            slam_system.process_image_rgbd(create_dummy_image(), create_dummy_depth_image(), timestamp)

        elif system_type == "mono_inertial":
            slam_system.process_image_mono_inertial(create_dummy_image(), timestamp, create_dummy_imu_data(timestamp))
        
        else:
            pytest.fail(f"Test case for system type '{system_type}' is not implemented.")

        # 3. Assert: Check a common post-condition
        assert slam_system.get_pose().shape == (4, 4)


    def test_reset_functionality_for_each_mode(self, slam_system_and_type):
        """
        Tests the reset functionality for each system type.
        """
        # 1. Arrange: Unpack the fixture and set up data
        slam_system, system_type = slam_system_and_type
        timestamp = 2000.0
        
        print(f"Testing reset on a '{system_type}' system.")
        
        # 2. Act (Part 1): Process a frame to give the system a state to reset.
        # This re-uses the same logic as the first test to ensure it works.
        if system_type == "mono":
            slam_system.process_image_mono(create_dummy_image(), timestamp)
        elif system_type == "stereo":
            slam_system.process_image_stereo(create_dummy_image(), create_dummy_image(), timestamp)
        elif system_type == "rgbd":
            slam_system.process_image_rgbd(create_dummy_image(), create_dummy_depth_image(), timestamp)
        elif system_type == "mono_inertial":
            slam_system.process_image_mono_inertial(create_dummy_image(), timestamp, create_dummy_imu_data(timestamp))
        
        initial_reset_count = slam_system.get_reset_count()

        # 3. Act (Part 2): Call the reset method
        slam_system.reset()

        # 4. Assert: Verify the results of the reset
        assert slam_system.get_reset_count() > initial_reset_count
        assert slam_system.was_map_reset() is True, "was_map_reset() should be true immediately after reset."
        assert slam_system.was_map_reset() is False, "was_map_reset() should be false after being checked once."
        # After a reset, the tracking state should revert to a non-OK state.
        assert slam_system.get_tracking_state() in [
            orbslam3.TrackingState.NOT_INITIALIZED,
            orbslam3.TrackingState.NO_IMAGES_YET
        ]



# --- DEPRECATED: Generic Processing Mode Tests ---
# @pytest.mark.parametrize("sensor_details", [
#     {"name": "mono", "sensor_enum": orbslam3.Sensor.MONOCULAR, "config": CONFIGS["mono"]},
#     {"name": "stereo", "sensor_enum": orbslam3.Sensor.STEREO, "config": CONFIGS["stereo"]},
#     {"name": "rgbd", "sensor_enum": orbslam3.Sensor.RGBD, "config": CONFIGS["rgbd"]},
#     {"name": "mono_inertial", "sensor_enum": orbslam3.Sensor.IMU_MONOCULAR, "config": CONFIGS["mono_inertial"]},
#     # TODO: Uncomment when stereo_inertial and rgbd_inertial are implemented
#     # {"name": "stereo_inertial", "sensor_enum": orbslam3.Sensor.IMU_STEREO, "config": CONFIGS["stereo_inertial"]},
#     # {"name": "rgbd_inertial", "sensor_enum": orbslam3.Sensor.IMU_RGBD, "config": CONFIGS["rgbd_inertial"]}
# ])
# class TestProcessingModes:
#     """Contains smoke tests for each sensor processing mode, using helper functions."""

#     def test_process_frame(self, sensor_details):
#         """A generic test to ensure each processing mode can be called without crashing."""
#         slam = orbslam3.System(VOCAB_FILE, sensor_details["config"], sensor_details["sensor_enum"])
#         slam.initialize()
        
#         timestamp = 2000.0
#         image_left = create_dummy_image()

#         try:
#             if sensor_details["name"] == "mono":
#                 slam.process_image_mono(image_left, timestamp)
#             elif sensor_details["name"] == "stereo":
#                 image_right = create_dummy_image() 
#                 slam.process_image_stereo(image_left, image_right, timestamp)
#             elif sensor_details["name"] == "rgbd":
#                 depth_img = create_dummy_image().astype(np.uint16) * 10 
#                 slam.process_image_rgbd(image_left, depth_img, timestamp)
#             elif sensor_details["name"] == "mono_inertial":
#                 imu_data = create_dummy_imu_data(timestamp)
#                 slam.process_image_mono_inertial(image_left, timestamp, imu_data)
#             # TODO: Uncomment when stereo_inertial and rgbd_inertial are implemented
#             # elif sensor_details["name"] == "stereo_inertial":
#             #     image_right = create_dummy_image()
#             #     imu_data = create_dummy_imu_data(timestamp)
#             #     slam.process_image_stereo_inertial(image_left, image_right, timestamp, imu_data)
#             # elif sensor_details["name"] == "rgbd_inertial":
#             #     depth_img = create_dummy_image().astype(np.uint16) * 10
#             #     imu_data = create_dummy_imu_data(timestamp)
#             #     slam.process_image_rgbd_inertial(image_left, depth_img, timestamp, imu_data)
            
#             pose = slam.get_pose()
#             assert pose.shape == (4, 4)

#         finally:
#             slam.shutdown()

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