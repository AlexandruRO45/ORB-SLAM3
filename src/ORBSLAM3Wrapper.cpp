/**
 * @file ORBSLAM3Wrapper.cpp
 * @brief pybind11 wrapper for ORB-SLAM3
 */

// ## 1. Includes and Namespace
// #############################################################################

#include <cmath>
#include <iostream>

// --- pybind11 Includes ---
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// --- ORB-SLAM3 Includes ---
#include <ORB_SLAM3_engine/include/Converter.h>
#include <ORB_SLAM3_engine/include/ImuTypes.h>
#include <ORB_SLAM3_engine/include/KeyFrame.h>
#include <ORB_SLAM3_engine/include/MapPoint.h>
#include <ORB_SLAM3_engine/include/Tracking.h>

// --- OpenCV Includes ---
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

// --- Local Includes ---
#include "NDArrayConverter.h"
#include "ORBSLAM3Wrapper.h"

namespace py = pybind11;

// ## 2. ORBSLAM3Python Class Implementation
// #############################################################################

// -----------------------------------------------------------------------------
// Section 2.1: Lifecycle Management
// -----------------------------------------------------------------------------

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
            std::cout << "\n Closing down ORB-SLAM3 system..." << std::endl;
        system->Shutdown();
    }
}

bool ORBSLAM3Python::initialize()
{
    system = std::make_unique<ORB_SLAM3::System>(vocabularyFile, settingsFile, sensorMode, bUseViewer);

    // Reset internal state flags
    mbFirstFrame = true;
    mbMapResetOccurred = false;
    mnResetCounter = 0;
    mLastTrackingState = ORB_SLAM3::Tracking::SYSTEM_NOT_READY;

    if (!system)
    {
        std::cerr << "\n Failed to initialize ORB-SLAM3 system!" << std::endl;
        return false;
    }
    return true;
}

void ORBSLAM3Python::shutdown()
{
    if (system)
    {
        if (m_bLoggingEnabled)
            std::cout << "\n Shutting down ORB-SLAM3 system..." << std::endl;
        system->Shutdown();
        system = nullptr;
    }
}

void ORBSLAM3Python::reset()
{
    if (system)
    {
        if (m_bLoggingEnabled)
            std::cout << "\n Resetting ORB-SLAM3 system..." << std::endl;
        system->Reset();
        mbMapResetOccurred = true;
        mnResetCounter++;
    }
}

// -----------------------------------------------------------------------------
// Section 2.2: Frame Processing
// -----------------------------------------------------------------------------

// bool ORBSLAM3Python::processFrame(const cv::Mat &image, const cv::Mat &rightImage, const cv::Mat &depthImage, const cv::Mat &mask, double timestamp, const std::vector<ORB_SLAM3::IMU::Point> &imuMeas)
// {
//     if (!system)
//         return false;

//     // The core ORB-SLAM3 Tracking object can accept a mask to ignore features in certain regions.
//     // We pass the mask to the tracker before processing the frame.
//     if (!mask.empty())
//     {
//         system->GetTracker()->SetMask(mask);
//     }

//     // Call the correct internal Track... function based on the sensor mode
//     // set during initialization.
//     switch (sensorMode)
//     {
//     case ORB_SLAM3::System::eSensor::MONOCULAR:
//         pose = system->TrackMonocular(image, timestamp);
//         break;
//     case ORB_SLAM3::System::eSensor::STEREO:
//         pose = system->TrackStereo(image, rightImage, timestamp);
//         break;
//     case ORB_SLAM3::System::eSensor::RGBD:
//         pose = system->TrackRGBD(image, depthImage, timestamp);
//         break;
//     case ORB_SLAM3::System::eSensor::IMU_MONOCULAR:
//         pose = system->TrackMonocular(image, timestamp, imuMeas);
//         break;
//     // TODO: ... add cases for IMU_STEREO and IMU_RGBD when ready
//     default:
//         std::cerr << "Unsupported sensor mode in processFrame!" << std::endl;
//         return false;
//     }

//     return this->postProcessFrame();
// }

bool ORBSLAM3Python::processMono(cv::Mat image, double timestamp)
{
    if (!system)
    {
        if (m_bLoggingEnabled)
            std::cout << "processMono - System not initialized!" << std::endl;
        return false;
    }
    if (image.empty())
    {
        if (m_bLoggingEnabled)
            std::cout << "processMono - Invalid image data!" << std::endl;
        return false;
    }

    pose = system->TrackMonocular(image, timestamp);
    return this->postProcessFrame();
}

bool ORBSLAM3Python::processStereo(cv::Mat leftImage, cv::Mat rightImage, double timestamp)
{
    if (!system)
    {
        if (m_bLoggingEnabled)
            std::cout << "processStereo - System not initialized!" << std::endl;
        return false;
    }
    if (leftImage.empty() || rightImage.empty())
    {
        if (m_bLoggingEnabled)
            std::cout << "processStereo - Invalid image data!" << std::endl;
        return false;
    }

    pose = system->TrackStereo(leftImage, rightImage, timestamp);
    return this->postProcessFrame();
}

bool ORBSLAM3Python::processRGBD(cv::Mat image, cv::Mat depthImage, double timestamp)
{
    if (!system)
    {
        if (m_bLoggingEnabled)
            std::cout << "processRGBD - System not initialized!" << std::endl;
        return false;
    }
    if (image.empty() || depthImage.empty())
    {
        if (m_bLoggingEnabled)
            std::cout << "processRGBD - Invalid image or depth data!" << std::endl;
        return false;
    }

    pose = system->TrackRGBD(image, depthImage, timestamp);
    return this->postProcessFrame();
}

bool ORBSLAM3Python::processMonoInertial(cv::Mat image, double timestamp, std::vector<ORB_SLAM3::IMU::Point> imuMeas)
{
    if (!system)
    {
        if (m_bLoggingEnabled)
            std::cout << "processMonoInertial - System not initialized!" << std::endl;
        return false;
    }
    if (image.empty())
    {
        if (m_bLoggingEnabled)
            std::cout << "processMonoInertial - Invalid image data!" << std::endl;
        return false;
    }

    pose = system->TrackMonocular(image, timestamp, imuMeas);
    return this->postProcessFrame();
}

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

// -----------------------------------------------------------------------------
// Section 2.3: Data & State Retrieval
// -----------------------------------------------------------------------------

ORB_SLAM3::Tracking::eTrackingState ORBSLAM3Python::getTrackingState() const
{
    if (!system)
        return ORB_SLAM3::Tracking::SYSTEM_NOT_READY;
    return static_cast<ORB_SLAM3::Tracking::eTrackingState>(system->GetTrackingState());
}

bool ORBSLAM3Python::isRunning() const
{
    return system != nullptr;
}

bool ORBSLAM3Python::isLost() const
{
    if (!system)
        return true;
    return system->isLost();
}

Eigen::Matrix4f ORBSLAM3Python::get_pose()
{
    return pose.matrix();
}

std::vector<Eigen::Matrix4f> ORBSLAM3Python::getTrajectory() const
{
    if (!system)
        return {};
    return system->GetCameraTrajectory();
}

py::array_t<short> ORBSLAM3Python::get2DOccMap() const
{
    auto map = system->Get2DOccMap();
    return py::array_t<short>({map.m_height, map.m_width}, {map.m_width * 2, 2}, &map.data.front());
}

std::tuple<std::vector<MapNodeData>, std::vector<MapEdgeData>> ORBSLAM3Python::getMapGraph()
{
    // Access the Atlas directly through the member pointer 'mpAtlas' in the ORB_SLAM3::System object.
    if (!system || !system->mpAtlas)
    {
        return {};
    }

    std::vector<MapNodeData> nodes;
    std::vector<MapEdgeData> edges;

    // Call the GetAllKeyFrames() method on the mpAtlas object.
    const std::vector<ORB_SLAM3::KeyFrame *> allKeyFrames = system->mpAtlas->GetAllKeyFrames();

    // Use a set for efficient lookup to ensure edges are only between keyframes in the current active map.
    const std::unordered_set<ORB_SLAM3::KeyFrame *> keyFrameSet(allKeyFrames.begin(), allKeyFrames.end());

    for (ORB_SLAM3::KeyFrame *pKF : allKeyFrames)
    {
        if (!pKF || pKF->isBad())
            continue;

        // 1. Create a MapNodeData object for each valid keyframe.
        // GetPoseInverse() provides the pose of the camera in world coordinates (Twc).
        nodes.push_back({(int)pKF->mnId,
                         pKF->GetPoseInverse().matrix(),
                         pKF->mTimeStamp});

        // 2. Create MapEdgeData for its connections from the covisibility graph.
        // GetCovisibilityConnectedKeyFrames() is a method on the KeyFrame object.
        const auto &connectedKeyFrames = pKF->GetCovisibilityConnectedKeyFrames();
        for (ORB_SLAM3::KeyFrame *pConnKF : connectedKeyFrames)
        {
            if (!pConnKF || pConnKF->isBad())
                continue;

            // To avoid duplicate edges (e.g., 1->2 and 2->1), only add an edge
            // if the current keyframe's ID is less than the connected one's.
            if (pKF->mnId < pConnKF->mnId && keyFrameSet.count(pConnKF))
            {
                edges.push_back({(int)pKF->mnId,
                                 (int)pConnKF->mnId,
                                 (float)pKF->GetCovisibilityWeight(pConnKF)});
            }
        }
    }
    return std::make_tuple(nodes, edges);
}

// -----------------------------------------------------------------------------
// Section 2.4: Map Reset Detection
// -----------------------------------------------------------------------------

bool ORBSLAM3Python::wasMapReset()
{
    if (!system)
        return false;

    bool resetDetected = false;

    // Check for explicit reset call
    if (mbMapResetOccurred)
    {
        if (m_bLoggingEnabled)
            std::cout << "Map reset detected: Explicit reset flag is set." << std::endl;
        resetDetected = true;
        mbMapResetOccurred = false; // Consume flag
        return true;
    }

    // Check for tracking state changes
    auto currentState = static_cast<ORB_SLAM3::Tracking::eTrackingState>(system->GetTrackingState());
    if (mLastTrackingState == ORB_SLAM3::Tracking::OK &&
        (currentState == ORB_SLAM3::Tracking::NOT_INITIALIZED || currentState == ORB_SLAM3::Tracking::RECENTLY_LOST))
    {
        resetDetected = true;
    }
    mLastTrackingState = currentState;

    // Check for large position jumps
    auto trajectory = getTrajectory();
    if (!trajectory.empty())
    {
        Eigen::Vector3f currentPosition = trajectory.back().block<3, 1>(0, 3);
        if (!mbFirstFrame)
        {
            float positionChange = (currentPosition - Eigen::Map<Eigen::Vector3f>(mvLastPosition.data())).norm();
            if (positionChange > m_fPositionJumpThreshold && system->GetTrackingState() != ORB_SLAM3::Tracking::LOST)
            {
                if (m_bLoggingEnabled)
                    std::cout << "Map reset detected: Large position jump of " << positionChange << "m" << std::endl;
                resetDetected = true;
            }
        }
        mvLastPosition[0] = currentPosition.x();
        mvLastPosition[1] = currentPosition.y();
        mvLastPosition[2] = currentPosition.z();
    }

    if (resetDetected)
        mnResetCounter++;
    return resetDetected;
}

int ORBSLAM3Python::getResetCount() const
{
    return mnResetCounter;
}

// -----------------------------------------------------------------------------
// Section 2.5: Configuration Setters
// -----------------------------------------------------------------------------

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

// ## 3. pybind11 Module Definition
// #############################################################################

PYBIND11_MODULE(_core, m)
{
    m.doc() = "Python bindings for the ORB-SLAM3 system";

    // Initialize the numpy <-> cv::Mat converter
    NDArrayConverter::init_numpy();

    // --- Enum Bindings ---
    py::enum_<ORB_SLAM3::System::eSensor>(m, "Sensor")
        .value("MONOCULAR", ORB_SLAM3::System::eSensor::MONOCULAR)
        .value("STEREO", ORB_SLAM3::System::eSensor::STEREO)
        .value("RGBD", ORB_SLAM3::System::eSensor::RGBD)
        .value("IMU_MONOCULAR", ORB_SLAM3::System::eSensor::IMU_MONOCULAR)
        .value("IMU_STEREO", ORB_SLAM3::System::eSensor::IMU_STEREO)
        .value("IMU_RGBD", ORB_SLAM3::System::eSensor::IMU_RGBD);

    py::enum_<ORB_SLAM3::Tracking::eTrackingState>(m, "TrackingState")
        .value("SYSTEM_NOT_READY", ORB_SLAM3::Tracking::eTrackingState::SYSTEM_NOT_READY)
        .value("NO_IMAGES_YET", ORB_SLAM3::Tracking::eTrackingState::NO_IMAGES_YET)
        .value("NOT_INITIALIZED", ORB_SLAM3::Tracking::eTrackingState::NOT_INITIALIZED)
        .value("OK", ORB_SLAM3::Tracking::eTrackingState::OK)
        .value("RECENTLY_LOST", ORB_SLAM3::Tracking::eTrackingState::RECENTLY_LOST)
        .value("LOST", ORB_SLAM3::Tracking::eTrackingState::LOST)
        .value("OK_KLT", ORB_SLAM3::Tracking::eTrackingState::OK_KLT);

    // --- IMU Struct Binding ---
    py::module_ imu_module = m.def_submodule("IMU", "IMU related classes");
    py::class_<ORB_SLAM3::IMU::Point>(imu_module, "Point")
        .def(py::init<const float &, const float &, const float &, const float &, const float &, const float &, const double &>(),
             py::arg("acc_x"), py::arg("acc_y"), py::arg("acc_z"),
             py::arg("ang_vel_x"), py::arg("ang_vel_y"), py::arg("ang_vel_z"),
             py::arg("timestamp"))
        .def_readonly("a", &ORB_SLAM3::IMU::Point::a)  // Accelerometer
        .def_readonly("w", &ORB_SLAM3::IMU::Point::w)  // Gyroscope
        .def_readonly("t", &ORB_SLAM3::IMU::Point::t); // Timestamp

    // --- Main Class Binding ---
    py::class_<ORBSLAM3Python>(m, "system")
        // Lifecycle
        .def(py::init<std::string, std::string, ORB_SLAM3::System::eSensor>(),
             py::arg("vocab_file"), py::arg("settings_file"), py::arg("sensor_type"))
        .def("initialize", &ORBSLAM3Python::initialize, "Initializes the SLAM system.")
        .def("shutdown", &ORBSLAM3Python::shutdown, "Shuts down the SLAM system.")
        .def("reset", &ORBSLAM3Python::reset, "Resets the map and tracking.")

        // Frame Processing
        .def("process_image_mono", &ORBSLAM3Python::processMono, py::arg("image"), py::arg("time_stamp"))
        .def("process_image_stereo", &ORBSLAM3Python::processStereo, py::arg("left_image"), py::arg("right_image"), py::arg("time_stamp"))
        .def("process_image_rgbd", &ORBSLAM3Python::processRGBD, py::arg("image"), py::arg("depth"), py::arg("time_stamp"))
        .def("process_image_mono_inertial", &ORBSLAM3Python::processMonoInertial, py::arg("image"), py::arg("time_stamp"), py::arg("imu_meas"))

        // Data & State Retrieval
        .def("is_running", &ORBSLAM3Python::isRunning)
        .def("is_lost", &ORBSLAM3Python::isLost)
        .def("get_tracking_state", &ORBSLAM3Python::getTrackingState)
        .def("get_pose", &ORBSLAM3Python::get_pose, "Returns pose as a 4x4 Eigen Matrix.")
        .def("get_trajectory", &ORBSLAM3Python::getTrajectory)
        .def("get_2d_occmap", &ORBSLAM3Python::get2DOccMap)

        // Map Reset Detection
        .def("was_map_reset", &ORBSLAM3Python::wasMapReset)
        .def("get_reset_count", &ORBSLAM3Python::getResetCount)

        // Configuration
        .def("set_use_viewer", &ORBSLAM3Python::setUseViewer)
        .def("set_logging", &ORBSLAM3Python::setLogging, py::arg("enabled"))
        .def("set_reset_jump_threshold", &ORBSLAM3Python::setResetJumpThreshold, py::arg("threshold"));
}