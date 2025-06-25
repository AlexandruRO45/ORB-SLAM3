#ifndef ORBSLAM3_WRAPPER_H
#define ORBSLAM3_WRAPPER_H

#include <memory>
#include <string>
#include <vector>
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
    bool isRunning();
    void setUseViewer(bool useViewer);
    std::vector<Eigen::Matrix4f> getTrajectory() const;
    int getTrackingState() const;
    bool isLost() const;
    bool wasMapReset();
    int getResetCount() const;
    auto get_pose(){
      return pose.matrix();
    }
    py::array_t<short> get2DOccMap() const;

  private:
    std::string vocabluaryFile;
    std::string settingsFile;
    ORB_SLAM3::System::eSensor sensorMode;
    std::shared_ptr<ORB_SLAM3::System> system;
    bool bUseViewer;
    bool mbMapResetOccurred;
    int mnResetCounter;
    bool mbFirstFrame;
    ORB_SLAM3::Tracking::eTrackingState mLastTrackingState; // Changed from int to proper enum type
    std::vector<float> mvLastPosition;
    bool bUseRGB;
    Sophus::SE3f pose;
};

#endif // ORBSLAM3_WRAPPER_H
