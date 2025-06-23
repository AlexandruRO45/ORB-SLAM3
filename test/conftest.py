import pytest
from pathlib import Path
import orbslam3 # Your package

# Store created systems to shut them down later
SLAM_SYSTEMS_TO_CLEANUP = []

@pytest.fixture(scope="session")
def slam_system_factory():
    """
    A factory fixture that returns a function to create and manage SLAM system instances.
    This allows tests to create systems with different sensor configurations.
    All created systems are automatically shut down at the end of the test session.
    """
    
    def _create_system(sensor_mode):
        """The actual factory function that will be returned to the tests."""
        project_root = Path(__file__).parent.parent
        
        # --- Define paths based on sensor mode ---
        if sensor_mode == orbslam3.Sensor.MONOCULAR:
            settings_file = project_root / "third_party" / "ORB_SLAM3_engine" / "Examples" / "Monocular" / "TUM1.yaml"
        elif sensor_mode == orbslam3.Sensor.STEREO:
            settings_file = project_root / "third_party" / "ORB_SLAM3_engine" / "Examples" / "Stereo" / "EuRoC.yaml"
        elif sensor_mode == orbslam3.Sensor.RGBD:
            settings_file = project_root / "third_party" / "ORB_SLAM3_engine" / "Examples" / "RGB-D" / "TUM1.yaml"
        else:
            raise ValueError(f"Unsupported sensor mode for testing: {sensor_mode}")

        vocab_path = project_root / "third_party" / "ORB_SLAM3_engine" / "Vocabulary" / "ORBvoc.txt"

        if not vocab_path.exists() or not settings_file.exists():
            raise FileNotFoundError(
                f"Ensure vocab ({vocab_path}) and settings ({settings_file}) files exist."
            )

        # In your C++ code, the class is exposed as 'system'
        slam = orbslam3.System(str(vocab_path), str(settings_file), sensor_mode)
        slam.set_use_viewer(False)
        slam.initialize()
        
        # Keep track of the system for cleanup
        SLAM_SYSTEMS_TO_CLEANUP.append(slam)
        return slam

    # --- Yield the factory function to the test session ---
    yield _create_system

    # --- Teardown code: runs after all tests are complete ---
    print(f"\n--- Shutting down {len(SLAM_SYSTEMS_TO_CLEANUP)} SLAM system(s) ---")
    for slam in SLAM_SYSTEMS_TO_CLEANUP:
        slam.shutdown()