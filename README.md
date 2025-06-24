# Python Bindings for ORB-SLAM3

[![PyPI version](https://img.shields.io/pypi/v/orbslam3.svg)](https://pypi.org/project/orbslam3/)
[![Build Status](https://github.com/alexandrusava/ORB-SLAM3/actions/workflows/build_test.yml/badge.svg)](https://github.com/alexandrusava/ORB-SLAM3/actions/workflows/build_test.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This package provides a more up-to-date, pre-compiled Python bindings from the original repo **ORB-SLAM3-PYTHON** that has been inactive for years.

This wrapper allows seamless integration of SLAM and Global Visual Odometry functionalities into Python applications without requiring manual compilation of the C++ source code on the end-user's machine.

## Key Features

* **Pre-compiled Wheels**: No need for a C++ toolchain on the user's machine. `pip install orbslam3` just works on Linux (tested), Windows, and macOS.
* **Numpy Integration**: Easily pass images and receive poses as `numpy.ndarray` objects.
* **Simple API**: A straightforward class-based interface for initializing the SLAM system and processing frames.
* **Supports Standard Datasets**: Comes with example configurations for popular datasets like TUM, KITTI, and EuRoC.

## System Dependencies

While the Python package is self-contained, ORB-SLAM3 relies on a few system-level libraries to function. You must install them using your system's package manager.

**On Debian/Ubuntu:**

```bash
sudo apt-get update
sudo apt-get install -y libopencv-dev libeigen3-dev
```


**On macOS (using Homebrew):**
```bash
brew install opencv eigen
```

## Installation

Once the system dependencies are installed, you can install the package from PyPI:
```bash
pip install orbslam3
```

<details>
  <summary>Original ORB-SLAM3-PYTHON readme (Click me)</summary>


ORB-SLAM3-PYTHON
===

Python bindings generated using [pybind11](https://pybind11.readthedocs.io/en/stable/). We use a modified version of ORB-SLAM3 (included as a submodule) to exntend interfaces. It might not be the most up-to-date with the original ORB-SLAM3.

## Update

+ Oct. 3rd, 2023: Added demo code.
+ Feb. 7th, 2023: First working version. 

## Dependancy

+ OpenCV >= 4.4
+ Pangolin
+ Eigen >= 3.1
+ C++11 or C++0x Compiler

## Installation

1. Clone the repo with `--recursive`
2. Install `Eigen`, `Pangolin` and `OpenCV` if you havn't already.
3. `ORB-SLAM3` requires `openssl` to be installed: `sudo apt-get install libssl-dev`
4. Run `python setup install` or `pip install .`.
5. Please raise an issue if you run into any.

## Demo

Please see the demo at `demo/run_rgb.py` for how to use this code. For example, you can run this demo with (by substituting the appropriate arguments):

```bash
python demo/run_rgbd.py \
    --vocab_file=third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt \
    --settings_file=third_party/ORB_SLAM3/Examples/RGB-D/TUM1.yaml \
    --dataset_path=/mnt/dataset2/TUM/rgbd_dataset_freiburg1_xyz
```

