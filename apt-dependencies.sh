#!/bin/bash
# This script installs the necessary dependencies for the project using apt-get.
set -e

sudo apt-get update && sudo apt-get upgrade -y
# Install essential build tools
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    libeigen3-dev \
    libopencv-dev \
    libboost-serialization-dev \
    liboctomap-dev \
    libssl-dev

