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

py::dict ORBSLAM3Python::get_current_pose()
{
    if (!system || pose.matrix().isIdentity())
    {
        return py::none();
    }

    Eigen::Quaternionf q(pose.rotationMatrix());
    Eigen::Vector3f t = pose.translation();

    py::dict p;
    // Return quaternion as [w, x, y, z] for standard Python libraries (e.g., scipy)
    py::array_t<float> orientation_arr(4);
    float *orientation_ptr = static_cast<float *>(orientation_arr.request().ptr);
    orientation_ptr[0] = q.w();
    orientation_ptr[1] = q.x();
    orientation_ptr[2] = q.y();
    orientation_ptr[3] = q.z();

    p["position"] = py::array_t<float>(3, t.data());
    p["orientation"] = orientation_arr;
    p["timestamp"] = system->GetLastTrackedFrameTimestamp();
    p["covariance"] = py::none();

    return p;
}

std::vector<Eigen::Matrix4f> ORBSLAM3Python::getTrajectory() const
{
    if (!system)
        return {};
    return system->GetCameraTrajectory();
}

py::dict ORBSLAM3Python::get_map_graph()
{
    if (!system)
        return py::dict();

    std::vector<ORB_SLAM3::KeyFrame *> vpKFs = system->GetAtlas()->GetAllKeyFrames();
    py::list nodes;
    py::list edges;

    for (ORB_SLAM3::KeyFrame *pKF : vpKFs)
    {
        if (!pKF || pKF->isBad())
            continue;

        // Create MapNode dictionary
        py::dict node;
        Sophus::SE3f Tcw = pKF->GetPose();
        Eigen::Quaternionf q(Tcw.rotationMatrix());

        py::dict pose_dict;
        pose_dict["position"] = py::array_t<float>(3, Tcw.translation().data());
        pose_dict["orientation"] = py::array_t<float>({q.w(), q.x(), q.y(), q.z()});
        pose_dict["timestamp"] = pKF->mTimeStamp;
        pose_dict["covariance"] = py::none();

        node["id"] = pKF->mnId;
        node["pose"] = pose_dict;
        node["timestamp"] = pKF->mTimeStamp;
        nodes.append(node);

        // Create MapEdge dictionaries
        for (ORB_SLAM3::KeyFrame *pConn : pKF->GetConnectedKeyFrames())
        {
            py::dict edge;
            edge["from_node"] = pKF->mnId;
            edge["to_node"] = pConn->mnId;
            edges.append(edge);
        }
    }

    py::dict map_graph;
    map_graph["nodes"] = nodes;
    map_graph["edges"] = edges;
    map_graph["timestamp"] = system->GetLastTrackedFrameTimestamp();
    return map_graph;
}

py::array_t<short> ORBSLAM3Python::get2DOccMap() const
{
    auto map = system->Get2DOccMap();
    return py::array_t<short>({map.m_height, map.m_width}, {map.m_width * 2, 2}, &map.data.front());
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
        .def("get_current_pose", &ORBSLAM3Python::get_current_pose, "Returns pose as a Python dictionary.")
        .def("get_trajectory", &ORBSLAM3Python::getTrajectory)
        .def("get_map_graph", &ORBSLAM3Python::get_map_graph, "Returns the full map graph as a dictionary.")
        .def("get_2d_occmap", &ORBSLAM3Python::get2DOccMap)

        // Map Reset Detection
        .def("was_map_reset", &ORBSLAM3Python::wasMapReset)
        .def("get_reset_count", &ORBSLAM3Python::getResetCount)

        // Configuration
        .def("set_use_viewer", &ORBSLAM3Python::setUseViewer)
        .def("set_logging", &ORBSLAM3Python::setLogging, py::arg("enabled"))
        .def("set_reset_jump_threshold", &ORBSLAM3Python::setResetJumpThreshold, py::arg("threshold"));
}