#ifndef ORBSLAM3_WRAPPER_H
#define ORBSLAM3_WRAPPER_H

// --- Standard Library Includes ---
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cmath>

// --- Eigen Includes ---
#include <Eigen/Core>
#include <sophus/se3.hpp>

// --- ORB-SLAM3 Includes ---
#include <ORB_SLAM3_engine/include/System.h>
#include <ORB_SLAM3_engine/include/Tracking.h>
#include <ORB_SLAM3_engine/include/KeyFrame.h>
#include <ORB_SLAM3_engine/include/Converter.h>
#include <ORB_SLAM3_engine/include/MapPoint.h>
#include <ORB_SLAM3_engine/include/ImuTypes.h>

// --- Pybind11 Includes ---
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

// --- OpenCV Includes ---
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace py = pybind11;

class ORBSLAM3Python
{
public:
  // --- Lifecycle and Initialization ---
  ORBSLAM3Python(std::string vocabFile, std::string settingsFile,
                 ORB_SLAM3::System::eSensor sensorMode);
  ~ORBSLAM3Python();

  bool initialize();
  void shutdown();
  void reset();
  bool isRunning();

  // --- Frame Processing ---
  bool processMono(cv::Mat image, double timestamp);
  bool processMonoInertial(cv::Mat image, double timestamp, std::vector<ORB_SLAM3::IMU::Point> imuMeas);
  bool processStereo(cv::Mat leftImage, cv::Mat rightImage, double timestamp);
  bool processStereoInertial(cv::Mat leftImage, cv::Mat rightImage, double timestamp, std::vector<ORB_SLAM3::IMU::Point> imuMeas);
  bool processRGBD(cv::Mat image, cv::Mat depthImage, double timestamp);
  bool processRGBDInertial(cv::Mat image, cv::Mat depthImage, double timestamp, std::vector<ORB_SLAM3::IMU::Point> imuMeas);

  // --- Configuration ---
  void setUseViewer(bool useViewer);
  void setLogging(bool enabled);
  void setResetJumpThreshold(float threshold);

  // --- Data and State Access ---
  std::vector<Eigen::Matrix4f> getTrajectory() const;
  ORB_SLAM3::Tracking::eTrackingState getTrackingState() const;
  bool isLost() const;
  bool wasMapReset();
  int getResetCount() const;
  py::array_t<short> get2DOccMap() const;

private:
  // --- Private Helper Functions ---
  bool postProcessFrame();

  // --- Core SLAM System Components ---
  std::unique_ptr<ORB_SLAM3::System> system;
  ORB_SLAM3::System::eSensor sensorMode;
  std::string vocabularyFile;
  std::string settingsFile;
  Sophus::SE3<float> pose;

  // --- Configuration Settings ---
  bool bUseViewer;
  bool m_bLoggingEnabled;
  float m_fPositionJumpThreshold;

  // --- Tracking State and History ---
  bool mbFirstFrame;
  ORB_SLAM3::Tracking::eTrackingState mLastTrackingState;
  std::vector<float> mvLastPosition;
  bool mbMapResetOccurred;
  int mnResetCounter;
};

#endif // ORBSLAM3_WRAPPER_H