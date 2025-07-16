// --- Project Headers ---
#include "ORBSLAM3Wrapper.h"
#include "NDArrayConverter.h"

namespace py = pybind11;





// #################################################################################################
// ##
// ##  1. CONSTRUCTOR & DESTRUCTOR
// ##
// #################################################################################################

ORBSLAM3Python::ORBSLAM3Python(std::string vocabFile, std::string settingsFile, ORB_SLAM3::System::eSensor sensorMode)
    : vocabularyFile(vocabFile),
      settingsFile(settingsFile),
      sensorMode(sensorMode),
      system(nullptr),
      bUseViewer(false),
      m_bLoggingEnabled(true),
      m_fPositionJumpThreshold(1.0f),
      mbMapResetOccurred(false),
      mnResetCounter(0),
      mbFirstFrame(true),
      mLastTrackingState(ORB_SLAM3::Tracking::SYSTEM_NOT_READY)
{
    mvLastPosition = {0.0f, 0.0f, 0.0f};
}

ORBSLAM3Python::~ORBSLAM3Python()
{
    if (system)
    {
        if (m_bLoggingEnabled)
        {
            std::cout << "\n[ORBSLAM3] Closing down system..." << std::endl;
        }
        system->Shutdown();
    }
}



// #################################################################################################
// ##
// ##  2. SYSTEM LIFECYCLE MANAGEMENT
// ##
// #################################################################################################

bool ORBSLAM3Python::initialize()
{
    system = std::make_unique<ORB_SLAM3::System>(vocabularyFile, settingsFile, sensorMode, bUseViewer);
    if (!system)
    {
        std::cerr << "\n[ORBSLAM3] FATAL: Failed to initialize system!" << std::endl;
        return false;
    }

    // Reset internal state flags
    mbFirstFrame = true;
    mbMapResetOccurred = false;
    mnResetCounter = 0;
    mLastTrackingState = ORB_SLAM3::Tracking::SYSTEM_NOT_READY;
    return true;
}

void ORBSLAM3Python::shutdown()
{
    if (system)
    {
        if (m_bLoggingEnabled)
        {
            std::cout << "\n[ORBSLAM3] Shutting down system..." << std::endl;
        }
        system->Shutdown();
        system = nullptr; // Release the unique_ptr
    }
}

void ORBSLAM3Python::reset()
{
    if (system)
    {
        if (m_bLoggingEnabled)
        {
            std::cout << "\n[ORBSLAM3] Resetting system..." << std::endl;
        }
        system->Reset();
        mbMapResetOccurred = true;
        mnResetCounter++;
    }
}

bool ORBSLAM3Python::isRunning()
{
    return system != nullptr;
}



// #################################################################################################
// ##
// ##  3. PRIVATE HELPER FUNCTIONS - FOR 'FRAME PROCESSING'
// ##
// #################################################################################################

bool ORBSLAM3Python::postProcessFrame()
{
    this->wasMapReset();
    if (mbFirstFrame)
    {
        mbFirstFrame = false;
        mbMapResetOccurred = false;
    }
    return !system->isLost();
}



// #################################################################################################
// ##
// ##  4. FRAME PROCESSING
// ##
// #################################################################################################

bool ORBSLAM3Python::processMono(cv::Mat image, double timestamp) 
{
    if (!system)
    {
        if (m_bLoggingEnabled)
            std::cout << "\n[ORBSLAM3] processMono - System not initialized!" << std::endl;
        return false;
    }
    if (image.data)
    {
        if (m_bLoggingEnabled)
            std::cout << "\n[ORBSLAM3] processMono - Processing frame at t=" << timestamp << std::endl;
        pose = system->TrackMonocular(image, timestamp);
        if (m_bLoggingEnabled) {
            std::cout << "\n[ORBSLAM3] processMono - Frame processed, pose: " << pose.matrix() << std::endl;
            std::cout << "\n[ORBSLAM3] processMono - TrackMonocular completed" << std::endl;
        }
        return this->postProcessFrame();
    }
    else
    {
        if (m_bLoggingEnabled)
            std::cout << "\n[ORBSLAM3] processMono - Invalid image data!" << std::endl;
        return false;
    }
}

bool ORBSLAM3Python::processMonoInertial(cv::Mat image, double timestamp, std::vector<ORB_SLAM3::IMU::Point> imuMeas) 
{
    if (!system)
    {
        if (m_bLoggingEnabled)
            std::cout << "\n[ORBSLAM3] processMonoInertial - System not initialized!" << std::endl;
        return false;
    }
    if (m_bLoggingEnabled)
        std::cout << "\n[ORBSLAM3] processMonoInertial - Processing frame at t=" << timestamp
                  << " with " << imuMeas.size() << " IMU measurements" << std::endl;

    // 2. Loop through the data and convert it to the C++ struct
    for (size_t i = 0; i < imuMeas.size(); i++)
    {
        if (m_bLoggingEnabled)
            std::cout << "\n[ORBSLAM3] processMonoInertial - IMU[" << i << "]: t=" << imuMeas[i].t
                      << ", acc=[" << imuMeas[i].a[0] << ", " << imuMeas[i].a[1] << ", " << imuMeas[i].a[2]
                      << "], angVel=[" << imuMeas[i].w[0] << ", " << imuMeas[i].w[1] << ", " << imuMeas[i].w[2] << "]" << std::endl;

        // Check for extreme values
        if (std::abs(imuMeas[i].w[0]) > 100 || std::abs(imuMeas[i].w[1]) > 100 || std::abs(imuMeas[i].w[2]) > 100)
        {
            std::cout << "\n[ORBSLAM3] WARNING: Extremely high angular velocity detected!" << std::endl;
        }

        if (std::abs(imuMeas[i].a[0]) > 100 || std::abs(imuMeas[i].a[1]) > 100 || std::abs(imuMeas[i].a[2]) > 100)
        {
            std::cout << "\n[ORBSLAM3] WARNING: Extremely high acceleration detected!" << std::endl;
        }
    }

    if (image.data)
    {
        if (m_bLoggingEnabled)
            std::cout << "\n[ORBSLAM3] processMonoInertial - Processing stereo frame at t=" << timestamp << std::endl;
        pose = system->TrackMonocular(image, timestamp, imuMeas);
        if (m_bLoggingEnabled) {
            std::cout << "\n[ORBSLAM3] processMonoInertial - TrackMonocular completed, pose: " << pose.matrix() << std::endl;
            std::cout << "\n[ORBSLAM3] processMonoInertial - TrackMonocular completed" << std::endl;
        }
        return this->postProcessFrame();
    }
    else
    {
        if (m_bLoggingEnabled)
            std::cout << "\n[ORBSLAM3] processMonoInertial - Invalid image data!" << std::endl;
        return false;
    }
}

bool ORBSLAM3Python::processStereo(cv::Mat leftImage, cv::Mat rightImage, double timestamp) 
{
    if (!system)
    {
        if (m_bLoggingEnabled)
            std::cout << "\n[ORBSLAM3] processStereo - System not initialized!" << std::endl;
        return false;
    }
    if (leftImage.data && rightImage.data)
    {
        if (m_bLoggingEnabled)
            std::cout << "\n[ORBSLAM3] processStereo - Processing stereo frame at t=" << timestamp << std::endl;
        pose = system->TrackStereo(leftImage, rightImage, timestamp);
        if (m_bLoggingEnabled) {
            std::cout << "\n[ORBSLAM3] processStereo - Frame processed, pose: " << pose.matrix() << std::endl;
            std::cout << "\n[ORBSLAM3] processStereo - TrackStereo completed" << std::endl;
        }
        return this->postProcessFrame();
    }
    else
    {
        if (m_bLoggingEnabled)
            std::cout << "\n[ORBSLAM3] processStereo - Invalid image data!" << std::endl;
        return false;
    }
}

bool ORBSLAM3Python::processRGBD(cv::Mat image, cv::Mat depthImage, double timestamp) 
{
    if (!system)
    {
        if (m_bLoggingEnabled)
            std::cout << "\n[ORBSLAM3] processRGBD - System not initialized!" << std::endl;
        return false;
    }
    if (image.data && depthImage.data)
    {
        if (m_bLoggingEnabled)
            std::cout << "\n[ORBSLAM3] processRGBD - Processing RGB-D frame at t=" << timestamp << std::endl;
        pose = system->TrackRGBD(image, depthImage, timestamp);
        if (m_bLoggingEnabled) {
            std::cout << "\n[ORBSLAM3] processRGBD - Frame processed, pose: " << pose.matrix() << std::endl;
            std::cout << "\n[ORBSLAM3] processRGBD - TrackRGBD completed" << std::endl;
        }
        return this->postProcessFrame();
    }
    else
    {
        if (m_bLoggingEnabled)
            std::cout << "\n[ORBSLAM3] processRGBD - Invalid image or depth data!" << std::endl;
        return false;
    }
}



// #################################################################################################
// ##
// ##  5. CONFIGURATION SETTERS
// ##
// #################################################################################################

void ORBSLAM3Python::setUseViewer(bool useViewer)
{
    bUseViewer = useViewer;
}

void ORBSLAM3Python::setLogging(bool enabled)
{
    m_bLoggingEnabled = enabled;
}

void ORBSLAM3Python::setResetJumpThreshold(float threshold)
{
    m_fPositionJumpThreshold = threshold;
}



// #################################################################################################
// ##
// ##  6. DATA AND STATE ACCESSORS
// ##
// #################################################################################################

py::array_t<short> ORBSLAM3Python::get2DOccMap() const {
    auto map = system->Get2DOccMap();
    if (m_bLoggingEnabled)
        std::cout << "\n[ORBSLAM3] get2DOccMap - Map dimensions: " << map.m_height << "x" << map.m_width << std::endl;
    return py::array_t<short>(
        {map.m_height, map.m_width}, //shape
        {map.m_width*2, 2}, //strides
        &map.data.front()
        );
}

std::vector<Eigen::Matrix4f> ORBSLAM3Python::getTrajectory() const {
    if (!system)
    {
        if (m_bLoggingEnabled)
            std::cout << "\n[ORBSLAM3] getTrajectory - System not initialized!" << std::endl;
        return std::vector<Eigen::Matrix4f>();
    }
    return system->GetCameraTrajectory();
}

ORB_SLAM3::Tracking::eTrackingState ORBSLAM3Python::getTrackingState() const {
    if (!system)
        return ORB_SLAM3::Tracking::SYSTEM_NOT_READY;
    if (m_bLoggingEnabled)
        std::cout << "\n[ORBSLAM3] getTrackingState - Current tracking state: " << system->GetTrackingState() << std::endl;
    int rawState = system->GetTrackingState();
    if (m_bLoggingEnabled)
        std::cout << "\n[ORBSLAM3] getTrackingState - Raw state value: " << rawState << std::endl;
    return static_cast<ORB_SLAM3::Tracking::eTrackingState>(rawState);
    // return static_cast<ORB_SLAM3::Tracking::eTrackingState>(system->GetTrackingState());
}

bool ORBSLAM3Python::isLost() const {
    if (!system)
        return true;
    if (m_bLoggingEnabled)
        std::cout << "\n[ORBSLAM3] isLost - Checking if system is lost: " << system->isLost() << std::endl;
    return system->isLost();
}

bool ORBSLAM3Python::wasMapReset()
{
    if (!system)
        return false;

    // Various heuristics to detect map reset
    bool resetDetected = false;

    // Check for an explicitly called reset
    if (mbMapResetOccurred)
    {
        if (m_bLoggingEnabled)
            std::cout << "\n[ORBSLAM3] Map reset detected: Explicit reset flag is set." << std::endl;
        resetDetected = true;
        mbMapResetOccurred = false; // Clear the flag after reporting
        return resetDetected;
    }

    // Method 1: Check for tracking state changes from OK to something else
    // Use explicit casting from int to enum type
    int rawState = system->GetTrackingState();
    ORB_SLAM3::Tracking::eTrackingState currentState = static_cast<ORB_SLAM3::Tracking::eTrackingState>(rawState);
    if (m_bLoggingEnabled)
        std::cout << "\n[ORBSLAM3] wasMapReset - Current tracking state: " << currentState << std::endl;

    if (mLastTrackingState == ORB_SLAM3::Tracking::OK &&
        (currentState == ORB_SLAM3::Tracking::NOT_INITIALIZED ||
         currentState == ORB_SLAM3::Tracking::RECENTLY_LOST))
    {
        std::cout << "\n[ORBSLAM3] Map reset detected: Tracking state changed from OK to "
                  << (currentState == ORB_SLAM3::Tracking::NOT_INITIALIZED ? "NOT_INITIALIZED" : "RECENTLY_LOST")
                  << std::endl;
        resetDetected = true;
    }

    mLastTrackingState = currentState;

    // Method 2: Check if there's a large position jump in trajectory
    auto trajectory = getTrajectory();
    if (!trajectory.empty())
    {
        Eigen::Matrix4f lastPose = trajectory.back();
        float x = lastPose(0, 3);
        float y = lastPose(1, 3);
        float z = lastPose(2, 3);
        if (m_bLoggingEnabled)
            std::cout << "\n[ORBSLAM3] wasMapReset - Last position: (" << x << ", " << y << ", " << z << ")" << std::endl;

        // Calculate position difference
        if (!mbFirstFrame)
        {
            float dx = x - mvLastPosition[0];
            float dy = y - mvLastPosition[1];
            float dz = z - mvLastPosition[2];
            float positionChange = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (m_bLoggingEnabled)
                std::cout << "\n[ORBSLAM3] wasMapReset - Position change since last frame: " << positionChange << " m" << std::endl;

            // If jump is too large, it's likely a reset occurred
            // Threshold depends on your camera motion patterns, adjust in meters as needed
            // For example, 1.0 m is a common threshold for many applications
            if (positionChange > m_fPositionJumpThreshold && system->GetTrackingState() != ORB_SLAM3::Tracking::LOST)
            {
                if (m_bLoggingEnabled)
                    std::cout << "\n[ORBSLAM3] Map reset detected: Large position jump of " << positionChange << " m" << std::endl;
                resetDetected = true;
            }
        }

        // Update last position
        mvLastPosition[0] = x;
        mvLastPosition[1] = y;
        mvLastPosition[2] = z;
    }

    // If reset detected, increment counter
    if (resetDetected)
    {
        if (m_bLoggingEnabled)
            std::cout << "\n[ORBSLAM3] wasMapReset - Incrementing reset counter." << std::endl;
        mnResetCounter++;
    }

    // Special case for first frame
    if (mbFirstFrame)
    {
        if (m_bLoggingEnabled)
            std::cout << "\n[ORBSLAM3] wasMapReset - First frame detected, resetting state." << std::endl;
        mbFirstFrame = false;
        return false;
    }

    return resetDetected;
}

int ORBSLAM3Python::getResetCount() const
{
    if (m_bLoggingEnabled)
        std::cout << "\n[ORBSLAM3] getResetCount - Current reset counter: " << mnResetCounter << std::endl;
    return mnResetCounter;
}



// #################################################################################################
// ##
// ##  7. PYTHON BINDINGS (PYBIND11)
// ##
// #################################################################################################

PYBIND11_MODULE(_core, m)
{
    // Initialize the custom converter for cv::Mat <-> numpy.ndarray
    NDArrayConverter::init_numpy();

    // --- Data Structure Bindings ---
    py::module_ imu = m.def_submodule("IMU", "IMU related classes and functions");
    py::class_<ORB_SLAM3::IMU::Point>(imu, "Point")
        .def(py::init<const float &, const float &, const float &,
                      const float &, const float &, const float &,
                      const double &>(),
             py::arg("acc_x"), py::arg("acc_y"), py::arg("acc_z"),
             py::arg("ang_vel_x"), py::arg("ang_vel_y"), py::arg("ang_vel_z"),
             py::arg("timestamp"))
        // Expose the IMU data members as Python properties
        .def_readonly("a", &ORB_SLAM3::IMU::Point::a) // accelerometer data
        .def_readonly("w", &ORB_SLAM3::IMU::Point::w) // gyroscope data
        .def_readonly("t", &ORB_SLAM3::IMU::Point::t) // timestamp
        // Also add some convenience getters for individual components
        .def_property_readonly("ax", [](const ORB_SLAM3::IMU::Point &p)
                               { return p.a[0]; })
        .def_property_readonly("ay", [](const ORB_SLAM3::IMU::Point &p)
                               { return p.a[1]; })
        .def_property_readonly("az", [](const ORB_SLAM3::IMU::Point &p)
                               { return p.a[2]; })
        .def_property_readonly("wx", [](const ORB_SLAM3::IMU::Point &p)
                               { return p.w[0]; })
        .def_property_readonly("wy", [](const ORB_SLAM3::IMU::Point &p)
                               { return p.w[1]; })
        .def_property_readonly("wz", [](const ORB_SLAM3::IMU::Point &p)
                               { return p.w[2]; });

    // --- Enum Bindings --
    py::enum_<ORB_SLAM3::Tracking::eTrackingState>(m, "TrackingState")
        .value("SYSTEM_NOT_READY", ORB_SLAM3::Tracking::eTrackingState::SYSTEM_NOT_READY)
        .value("NO_IMAGES_YET", ORB_SLAM3::Tracking::eTrackingState::NO_IMAGES_YET)
        .value("NOT_INITIALIZED", ORB_SLAM3::Tracking::eTrackingState::NOT_INITIALIZED)
        .value("OK", ORB_SLAM3::Tracking::eTrackingState::OK)
        .value("RECENTLY_LOST", ORB_SLAM3::Tracking::eTrackingState::RECENTLY_LOST)
        .value("LOST", ORB_SLAM3::Tracking::eTrackingState::LOST)
        .value("OK_KLT", ORB_SLAM3::Tracking::eTrackingState::OK_KLT);

    py::enum_<ORB_SLAM3::System::eSensor>(m, "Sensor")
        .value("MONOCULAR", ORB_SLAM3::System::eSensor::MONOCULAR)
        .value("STEREO", ORB_SLAM3::System::eSensor::STEREO)
        .value("RGBD", ORB_SLAM3::System::eSensor::RGBD)
        .value("IMU_MONOCULAR", ORB_SLAM3::System::eSensor::IMU_MONOCULAR)
        .value("IMU_STEREO", ORB_SLAM3::System::eSensor::IMU_STEREO)
        .value("IMU_RGBD", ORB_SLAM3::System::eSensor::IMU_RGBD);


    // --- Main Wrapper Class Binding ---
    py::class_<ORBSLAM3Python>(m, "system")
        .def(py::init<std::string, std::string, ORB_SLAM3::System::eSensor>(), py::arg("vocab_file"), py::arg("settings_file"), py::arg("sensor_type"))
        // Lifecycle
        .def("initialize", &ORBSLAM3Python::initialize, "Initializes the SLAM system.")
        .def("is_running", &ORBSLAM3Python::isRunning, "Checks if the SLAM system is initialized and running.")
        .def("reset", &ORBSLAM3Python::reset, "Resets the map and tracking.")
        .def("shutdown", &ORBSLAM3Python::shutdown, "Shuts down the SLAM system and all threads.")
        // Processing
        .def("process_image_mono", &ORBSLAM3Python::processMono, py::arg("image"), py::arg("time_stamp"))
        .def("process_image_mono_inertial", &ORBSLAM3Python::processMonoInertial, py::arg("image"), py::arg("time_stamp"), py::arg("imu_meas"))
        .def("process_image_stereo", &ORBSLAM3Python::processStereo, py::arg("left_image"), py::arg("right_image"), py::arg("time_stamp"))
        .def("process_image_rgbd", &ORBSLAM3Python::processRGBD, py::arg("image"), py::arg("depth"), py::arg("time_stamp"))
        // Configuration
        .def("set_use_viewer", &ORBSLAM3Python::setUseViewer, py::arg("use_viewer"), "Enable/disable the viewer before initialization.")
        .def("set_logging", &ORBSLAM3Python::setLogging, py::arg("enabled"), "Enable/disable console logging.")
        .def("set_reset_jump_threshold", &ORBSLAM3Python::setResetJumpThreshold, py::arg("threshold"), "Set the position jump distance (in meters) to detect a map reset.")
        // Data Access
        .def("get_trajectory", &ORBSLAM3Python::getTrajectory, "Get the full camera trajectory.")
        .def("get_tracking_state", &ORBSLAM3Python::getTrackingState, "Get the current tracking state.")
        .def("is_lost", &ORBSLAM3Python::isLost, "Check if the system is in a LOST state.")
        .def("was_map_reset", &ORBSLAM3Python::wasMapReset, "Check if a map reset occurred in the last frame.")
        .def("get_reset_count", &ORBSLAM3Python::getResetCount, "Get the total number of map resets detected.")
        .def("get_2d_occ_map", &ORBSLAM3Python::get2DOccMap, "Get the 2D occupancy map.");
}
