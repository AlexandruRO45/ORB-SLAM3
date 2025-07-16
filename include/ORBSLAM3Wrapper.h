#ifndef ORBSLAM3_WRAPPER_H
#define ORBSLAM3_WRAPPER_H

/**
 * @file ORBSLAM3Wrapper.h
 * @brief Header for the pybind11 wrapper for ORB-SLAM3
 */

// ## 1. Includes
// #############################################################################

#include <memory>
#include <string>
#include <vector>

// --- Third-party Includes ---
#include <Eigen/Core>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// --- Forward Declarations for ORB-SLAM3 ---
namespace ORB_SLAM3
{
  class System;
  namespace IMU
  {
    struct Point;
  }
  namespace Tracking
  {
    enum eTrackingState : int;
  }
} // namespace ORB_SLAM3

namespace cv
{
  class Mat;
} // namespace OpenCV

// --- A simple struct to hold data for a MapNode ---
struct MapNodeData
{
  int id;
  Eigen::Matrix4f pose;
  double timestamp;
};

// A simple struct to hold data for a MapEdge
struct MapEdgeData
{
  int from_node_id;
  int to_node_id;
  float weight;
};

// Use pybind11 namespace
namespace py = pybind11;

// ## 2. ORBSLAM3Python Class Definition
// #############################################################################

class ORBSLAM3Python
{
public:
  // -------------------------------------------------------------------------
  // Section 2.1: Lifecycle Management
  // -------------------------------------------------------------------------

  /**
   * @brief Constructor for the ORBSLAM3Python wrapper.
   * @param vocabFile Path to the vocabulary file.
   * @param settingsFile Path to the settings YAML file.
   * @param sensorMode The sensor configuration to use.
   */
  ORBSLAM3Python(std::string vocabFile, std::string settingsFile,
                 ORB_SLAM3::System::eSensor sensorMode);

  /**
   * @brief Destructor. Ensures the SLAM system is shut down properly.
   */
  ~ORBSLAM3Python();

  // -------------------------------------------------------------------------
  // Section 2.2: System Control
  // -------------------------------------------------------------------------

  /**
   * @brief Initializes the underlying ORB-SLAM3 system.
   * @return True if initialization was successful, false otherwise.
   */
  bool initialize();

  /**
   * @brief Shuts down all SLAM threads and cleans up resources.
   */
  void shutdown();

  /**
   * @brief Resets the SLAM system to clear the map and start over.
   */
  void reset();

  /**
   * @brief Checks if the SLAM system is currently running.
   * @return True if the system is initialized and running.
   */
  bool isRunning();

  // -------------------------------------------------------------------------
  // Section 2.3: Frame Processing
  // -------------------------------------------------------------------------

  // bool processFrame(const cv::Mat &image,
  //                   const cv::Mat &rightImage, // Can be empty for Mono/RGBD
  //                   const cv::Mat &depthImage, // Can be empty for Mono/Stereo
  //                   const cv::Mat &mask,       // Can be empty if no filtering
  //                   double timestamp,
  //                   const std::vector<ORB_SLAM3::IMU::Point> &imuMeas);
  bool processMono(cv::Mat image, double timestamp);
  bool processStereo(cv::Mat leftImage, cv::Mat rightImage, double timestamp);
  bool processRGBD(cv::Mat image, cv::Mat depthImage, double timestamp);
  bool processMonoInertial(cv::Mat image, double timestamp, std::vector<ORB_SLAM3::IMU::Point> imuMeas);

  // -------------------------------------------------------------------------
  // Section 2.4: Data & State Retrieval
  // -------------------------------------------------------------------------

  ORB_SLAM3::Tracking::eTrackingState getTrackingState() const;
  bool isLost() const;
  Eigen::Matrix4f get_pose();
  std::vector<Eigen::Matrix4f> getTrajectory() const;
  py::array_t<short> get2DOccMap() const;
  std::tuple<std::vector<MapNodeData>, std::vector<MapEdgeData>> getMapGraph();

  // -------------------------------------------------------------------------
  // Section 2.5: Map Reset Detection
  // -------------------------------------------------------------------------

  bool wasMapReset();
  int getResetCount() const;

  // -------------------------------------------------------------------------
  // Section 2.6: Configuration Setters
  // -------------------------------------------------------------------------

  void setUseViewer(bool useViewer);
  void setLogging(bool enabled);
  void setResetJumpThreshold(float threshold);

private:
  // -------------------------------------------------------------------------
  // Section 2.7: Private Members
  // -------------------------------------------------------------------------

  // --- Private Helper Functions ---
  bool postProcessFrame();

  // --- SLAM System Members ---
  std::unique_ptr<ORB_SLAM3::System> system;
  Sophus::SE3f pose;

  // --- Configuration Members ---
  std::string vocabularyFile;
  std::string settingsFile;
  ORB_SLAM3::System::eSensor sensorMode;
  bool bUseViewer;
  bool m_bLoggingEnabled;
  float m_fPositionJumpThreshold;

  // --- State Tracking Members ---
  bool mbFirstFrame;
  bool mbMapResetOccurred;
  int mnResetCounter;
  ORB_SLAM3::Tracking::eTrackingState mLastTrackingState;
  std::vector<float> mvLastPosition;
};

#endif // ORBSLAM3_WRAPPER_H