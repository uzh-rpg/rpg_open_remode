# This file is part of REMODE - REgularized MOnocular Depth Estimation.
#
# Copyright (C) 2014 Matia Pizzoli <matia dot pizzoli at gmail dot com>
# Robotics and Perception Group, University of Zurich, Switzerland
# http://rpg.ifi.uzh.ch
#
# REMODE is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or any later version.
#
# REMODE is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

SET(PROJECT_NAME remode)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)
PROJECT(${PROJECT_NAME})
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")
SET(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
IF(CMAKE_COMPILER_IS_GNUCC)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
ELSE()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ENDIF()

SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_30,code=sm_30; --ptxas-options=-v)
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
find_package(CUDA REQUIRED)

## System dependencies are found with CMake's conventions
find_package(OpenCV REQUIRED)
find_package(Eigen REQUIRED)
find_package(Boost REQUIRED)

## Specify additional locations of header files
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
)

if(RMD_DEBUG)
  ADD_DEFINITIONS(-DRMD_DEBUG)
endif()

# Maximum extent of epipolar line search in pixels
ADD_DEFINITIONS(-DRMD_MAX_EXTENT_EPIPOLAR_SEARCH=100)

CUDA_ADD_LIBRARY(
    remode
    src/seed_matrix.cu
    src/depthmap_denoiser.cu
)

CUDA_ADD_LIBRARY(
    remodetest
    test/copy.cu
    test/sobel.cu
)

## Declare a cpp executable
add_executable(
    dataset_main
    src/depthmap.cpp
    src/test/dataset_main.cpp
    test/dataset.cpp
)

## Specify libraries to link a library or executable target against
target_link_libraries(
    dataset_main
    remode
    boost_system
    boost_filesystem
    ${OpenCV_LIBS}
)

# GTest
enable_testing()
find_package(GTest REQUIRED)
# Accepts the following variable as input:
# GTEST_ROOT - (as a CMake or environment variable)
#              The root directory of the gtest install prefix

include_directories(${GTEST_INCLUDE_DIRS})

add_executable(all_tests
  test/device_image_test.cpp
  test/seed_matrix_test.cpp
  test/main_test.cpp
  test/dataset.cpp
  test/epipolar_test.cpp
)

target_link_libraries(
  all_tests
  remodetest
  remode
  ${GTEST_BOTH_LIBRARIES}
  ${OpenCV_LIBS}
  pthread
  boost_system
  boost_filesystem)

# The following is to run make test
add_test(
  RMDCuTests
  all_tests --gtest_repeat=2
)
