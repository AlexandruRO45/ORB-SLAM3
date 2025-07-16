#ifndef ORBSLAM3_WRAPPER_H
#define ORBSLAM3_WRAPPER_H

#include <memory>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <ORB_SLAM3_engine/include/System.h>
#include <ORB_SLAM3_engine/include/Tracking.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

class ORBSLAM3Python
{
public:
  ORBSLAM3Python(std::string vocabFile, std::string settingsFile,
                 ORB_SLAM3::System::eSensor sensorMode);
  ~ORBSLAM3Python();

  bool initialize();
  bool isRunning();
  bool processMono(cv::Mat image, double timestamp);
  bool processMonoInertial(cv::Mat image, double timestamp, std::vector<ORB_SLAM3::IMU::Point> imuMeas);
  bool processStereo(cv::Mat leftImage, cv::Mat rightImage, double timestamp);
  bool processRGBD(cv::Mat image, cv::Mat depthImage, double timestamp);
  void reset();
  void shutdown();

  // --- Configuration Setters ---
  void setUseViewer(bool useViewer);
  void setLogging(bool enabled);               
  void setResetJumpThreshold(float threshold); 

  // --- Data Getters ---
  std::vector<Eigen::Matrix4f> getTrajectory() const;
  ORB_SLAM3::Tracking::eTrackingState getTrackingState() const;
  bool isLost() const;
  bool wasMapReset();
  int getResetCount() const;
  Eigen::Matrix4f get_pose(); // Note: Cannot be const as it accesses non-const pose member
  py::array_t<short> get2DOccMap() const;

private:
  // --- Private Helper Functions ---
  bool postProcessFrame(); 

  // --- SLAM System Members ---
  std::string vocabularyFile; 
  std::string settingsFile;
  ORB_SLAM3::System::eSensor sensorMode;
  std::unique_ptr<ORB_SLAM3::System> system;
  Sophus::SE3f pose;

  // --- Configuration and State Members ---
  bool bUseViewer;
  bool m_bLoggingEnabled;         
  float m_fPositionJumpThreshold; 

  // --- Reset Tracking Members ---
  bool mbMapResetOccurred;
  int mnResetCounter;
  bool mbFirstFrame;
  ORB_SLAM3::Tracking::eTrackingState mLastTrackingState;
  std::vector<float> mvLastPosition;
};

#endif // ORBSLAM3_WRAPPER_H
