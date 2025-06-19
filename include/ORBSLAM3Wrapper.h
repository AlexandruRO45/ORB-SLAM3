#ifndef ORB_SLAM3_PYTHON_H
#define ORB_SLAM3_PYTHON_H

#include <memory>
#include <System.h>
#include <Tracking.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

class ORBSLAM3Python
{
public:
    ORBSLAM3Python(std::string vocabFile, std::string settingsFile,
                   ORB_SLAM3::System::eSensor sensorMode = ORB_SLAM3::System::eSensor::RGBD);
    ~ORBSLAM3Python();

    bool initialize();
    bool processMono(cv::Mat image, double timestamp);
    bool processStereo(cv::Mat leftImage, cv::Mat rightImage, double timestamp);
    bool processRGBD(cv::Mat image, cv::Mat depthImage, double timestamp);
    void reset();
    void shutdown();
    bool isRunning();
    void setUseViewer(bool useViewer);
    std::vector<Eigen::Matrix4f> getTrajectory() const;
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
    bool bUseRGB;
    Sophus::SE3f pose;
};

#endif // ORB_SLAM3_PYTHON_H
