cmake_minimum_required(VERSION 3.4...3.18)
project(orbslam3)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "Using C++ standard: ${CMAKE_CXX_STANDARD}")

LIST(APPEND CMAKE_MODULE_PATH 
    ${PROJECT_SOURCE_DIR}/cmake
)

add_subdirectory(third_party/pybind11)
add_subdirectory(third_party/ORB_SLAM3_engine)

pybind11_add_module(_core 
    src/ORBSLAM3Wrapper.cpp
    src/NDArrayConverter.cpp
)

set_target_properties(_core PROPERTIES PREFIX "")

target_link_libraries(_core
PRIVATE
    ORB_SLAM3
)

target_compile_definitions(_core 
PRIVATE 
    VERSION_INFO=${EXAMPLE_VERSION_INFO}
)

target_include_directories(_core
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/ORB_SLAM3_engine>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/ORB_SLAM3_engine/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/ORB_SLAM3_engine/include/CameraModels>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/ORB_SLAM3/Thirdparty/g2o/g2o>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/ORB_SLAM3/Thirdparty/DBoW2/DBoW2>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/ORB_SLAM3/Thirdparty/Sophus/sophus>
)