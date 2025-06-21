import os
import subprocess
import sys
import tarfile
import numpy as np
# Make sure to import find_packages
from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir="", **kwargs):
        Extension.__init__(self, name, sources=[], **kwargs)
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(
            self.get_ext_fullpath(ext.name)))
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
            
        # Extract ORBvoc.txt.tar.gz before CMake (or after, if CMake moves it)
        vocab_src = os.path.join(ext.sourcedir, "third_party", "ORB_SLAM3_engine", "Vocabulary", "ORBvoc.txt.tar.gz")
        vocab_dst_dir = os.path.join(ext.sourcedir, "third_party", "ORB_SLAM3_engine", "Vocabulary")

        if os.path.exists(vocab_src):
            with tarfile.open(vocab_src, "r:gz") as tar:
                tar.extractall(path=vocab_dst_dir)
                print(f"Extracted ORBvoc.txt to {vocab_dst_dir}")

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_CXX_FLAGS='-w'",
            f"-DCMAKE_CXX_FLAGS='-I {np.get_include()}'"
        ]

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(
            ["cmake", "--build", ".", "-j8"], cwd=self.build_temp)


setup(
    name="orbslam3",
    version="1.2.5",
    description='SLAM and Global VO module for VNAV project',
    long_description="This package provides Python bindings for the ORB-SLAM3 visual SLAM system, allowing users to integrate SLAM functionalities into Python applications.",
    maintainer='Alex S.',
    maintainer_email='savaalexandru562@gmail.com',
    license='TODO: License declaration',
    url='https://github.com/AlexandruRO45/ORB-SLAM3',
    packages=find_packages(),
    package_dir={'': '.'},
    package_data={
        'orbslam3': ['*.so', '*.dll', '*.dylib'],  # Include shared libraries
    },
    include_package_data=True,
    install_requires=["numpy"],
    ext_modules=[CMakeExtension("orbslam3")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    extras_require={"test": ["pytest>=6.0"]},
    python_requires=">=3.8",
)
