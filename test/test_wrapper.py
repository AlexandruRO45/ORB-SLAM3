import pytest
import numpy as np
import orbslam3

# The slam_system_factory fixture is provided automatically by conftest.py

def test_enums_exist():
    """
    Tests that the enums from C++ are correctly exposed. 
    """
    assert hasattr(orbslam3, "Sensor")
    assert hasattr(orbslam3.Sensor, "MONOCULAR")
    assert hasattr(orbslam3.Sensor, "STEREO")
    assert hasattr(orbslam3.Sensor, "RGBD")

    assert hasattr(orbslam3, "TrackingState")
    assert hasattr(orbslam3.TrackingState, "OK")
    assert hasattr(orbslam3.TrackingState, "LOST")


class TestSystemAPI:
    """
    Test all methods of the 'system' class exposed from C++.
    """

    def test_initialization_and_running(self, slam_system_factory):
        """
        Tests the __init__, initialize(), and is_running() methods. 
        """
        slam = slam_system_factory(orbslam3.Sensor.MONOCULAR)
        # The is_running() method checks if the system pointer is not null. 
        assert slam.is_running()

    def test_set_use_viewer(self, slam_system_factory):
        """
        Tests that calling set_use_viewer() doesn't crash. 
        """
        slam = slam_system_factory(orbslam3.Sensor.MONOCULAR)
        try:
            slam.set_use_viewer(True)
            slam.set_use_viewer(False)
        except Exception as e:
            pytest.fail(f"set_use_viewer raised an exception: {e}")

    def test_get_pose(self, slam_system_factory):
        """
        Tests get_pose() returns a valid 4x4 numpy array. 
        """
        slam = slam_system_factory(orbslam3.Sensor.MONOCULAR)
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # process_image_mono is bound in C++. 
        slam.process_image_mono(dummy_image, 1.0)
        
        pose = slam.get_pose()
        assert isinstance(pose, np.ndarray)
        assert pose.shape == (4, 4)

    def test_get_trajectory(self, slam_system_factory):
        """
        Tests get_trajectory() returns a list of poses. 
        """
        slam = slam_system_factory(orbslam3.Sensor.MONOCULAR)
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        slam.process_image_mono(dummy_image, 1.0)
        slam.process_image_mono(dummy_image, 2.0)
        
        trajectory = slam.get_trajectory()
        assert isinstance(trajectory, list)
        # After two frames, we should have keyframes and a trajectory
        assert len(trajectory) > 0
        assert all(isinstance(p, np.ndarray) and p.shape == (4, 4) for p in trajectory)

    def test_reset(self, slam_system_factory):
        """
        Tests that reset() clears the trajectory. 
        """
        slam = slam_system_factory(orbslam3.Sensor.MONOCULAR)
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        slam.process_image_mono(dummy_image, 1.0)
        slam.process_image_mono(dummy_image, 2.0)
        
        # Ensure we have a trajectory before reset
        trajectory_before = slam.get_trajectory()
        assert len(trajectory_before) > 0

        # Now reset the system
        slam.reset()
        
        # After reset, the trajectory should be empty
        trajectory_after = slam.get_trajectory()
        assert isinstance(trajectory_after, list)
        assert len(trajectory_after) == 0

    def test_get_2d_occmap(self, slam_system_factory):
        """
        Tests get_2d_occmap() returns a 2D numpy array of dtype short. 
        """
        slam = slam_system_factory(orbslam3.Sensor.MONOCULAR)
        # You might need to process some frames for the map to be meaningful
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        slam.process_image_mono(dummy_image, 1.0)
        
        occ_map = slam.get_2d_occmap()
        assert isinstance(occ_map, np.ndarray)
        assert occ_map.ndim == 2
        assert occ_map.dtype == np.int16

    def test_shutdown(self, slam_system_factory):
        """
        Tests that the explicit shutdown() method can be called without error. 
        The factory fixture ensures cleanup, but we test the public API here.
        """
        slam = slam_system_factory(orbslam3.Sensor.MONOCULAR)
        assert slam.is_running()
        try:
            # We call shutdown explicitly here for the test
            slam.shutdown()
        except Exception as e:
            pytest.fail(f"shutdown() raised an exception: {e}")
        # Note: The 'slam' object itself still exists, but the internal threads are stopped.