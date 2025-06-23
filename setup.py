import os
import subprocess
import sys
import tarfile
import numpy as np
from pathlib import Path
from setuptools import Extension, setup, find_packages 
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """A custom extension for CMake-based projects."""
    def __init__(self, name: str, sourcedir: str = "", **kwargs):
        super().__init__(name, sources=[], **kwargs)
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Custom build_ext command to run CMake."""
    def build_extension(self, ext: CMakeExtension):
        import numpy as np # Import numpy here, as it's a build dependency

        extdir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()

        # Extract vocabulary file
        vocab_src = Path(ext.sourcedir) / "third_party" / "ORB_SLAM3_engine" / "Vocabulary" / "ORBvoc.txt.tar.gz"
        vocab_dst_dir = vocab_src.parent
        if vocab_src.exists():
            print(f"Extracting {vocab_src} to {vocab_dst_dir}")
            with tarfile.open(vocab_src, "r:gz") as tar:
                tar.extractall(path=vocab_dst_dir)

        # Allow user to override build type with an environment variable
        build_type = os.environ.get("CMAKE_BUILD_TYPE", "Release")
        
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DCMAKE_CXX_FLAGS=-I{np.get_include()}",
            "-Wno-dev" # Suppress CMake developer warnings
        ]

        # Allow user to pass extra CMake args
        if "CMAKE_ARGS" in os.environ:
            cmake_args += os.environ["CMAKE_ARGS"].split()
            
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        print(f"Configuring CMake project with: {' '.join(cmake_args)}")
        subprocess.check_call(["cmake", str(ext.sourcedir)] + cmake_args, cwd=build_temp)

        # Allow user to override parallel jobs 
        # TODO: Known bug if there are more than 4 jobs, CMake will fail to build
        build_jobs = str(4) # Default to 4 jobs for now, can be overridden by user
        # Uncomment the following lines to allow user to set this via environment variable
        # build_jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL")
        # if not build_jobs:
        #     build_jobs = str(os.cpu_count() or 4) # Default to number of CPUs
        
        print(f"Building project with {build_jobs} parallel jobs")
        subprocess.check_call(
            ["cmake", "--build", ".", "--parallel", build_jobs],
            cwd=build_temp
        )

setup(
    name="orbslam3",
    version="1.2.9",
    description='SLAM and Global VO module for VNAV project',
    long_description="This package provides Python bindings for the ORB-SLAM3 visual SLAM system, allowing users to integrate SLAM functionalities into Python applications.",
    packages=find_packages(),
    package_dir={'': '.'},
    install_requires=["numpy"],
    ext_modules=[CMakeExtension("orbslam3._core")], 
    cmdclass={"build_ext": CMakeBuild},
    package_data={
        'orbslam3': ['*.so', '*.pyd', '*.dylib'], 
    },
    include_package_data=True,
    zip_safe=False,
    extras_require={"test": ["pytest>=6.0"]},
    python_requires=">=3.8",
)