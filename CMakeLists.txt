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

SET(PROJECT_NAME rpg_open_remode)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)
PROJECT(${PROJECT_NAME})
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")
SET(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo

find_package(CUDA REQUIRED)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -use_fast_math)

# Specify compute capability
#list(APPEND CUDA_NVCC_FLAGS -arch=sm_30)
#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)

SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
IF(CMAKE_COMPILER_IS_GNUCC)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
ELSE()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ENDIF()

option(BUILD_ROS_NODE "Build ROS node" ON)
if(BUILD_ROS_NODE)
  ADD_DEFINITIONS(-DBUILD_ROS_NODE)
endif(BUILD_ROS_NODE)

option(BUILD_TESTS "Build test programs" ON)
if(BUILD_TESTS)
  ADD_DEFINITIONS(-DRMD_BUILD_TESTS)
endif(BUILD_TESTS)

# NCC patch size (must be odd integer, typical values: 3, 5, 7)
ADD_DEFINITIONS(-DRMD_CORR_PATCH_SIDE=5)
# Maximum extent of epipolar line search in pixels
ADD_DEFINITIONS(-DRMD_MAX_EXTENT_EPIPOLAR_SEARCH=100)

find_package(OpenCV REQUIRED)
find_package(Eigen REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem)

if(BUILD_ROS_NODE)
  find_package(catkin REQUIRED COMPONENTS
    roscpp roslib cmake_modules svo_msgs cv_bridge image_transport vikit_ros pcl_ros)
  catkin_package(
      DEPENDS OpenCV Eigen Boost
      CATKIN_DEPENDS roscpp roslib svo_msgs image_transport vikit_ros pcl_ros
      INCLUDE_DIRS include
      LIBRARIES rpg_open_remode rpg_open_remode_cuda
  )
else()
  set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
  set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
endif(BUILD_ROS_NODE)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

# Set link libraries
list(APPEND LINK_LIBS
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
)

if(BUILD_ROS_NODE)
  include_directories(${catkin_INCLUDE_DIRS})
  list(APPEND LINK_LIBS ${catkin_LIBRARIES})
endif(BUILD_ROS_NODE)

# Set CUDA sourcefiles
list(APPEND CU_SOURCE_FILES
    src/check_cuda_device.cu
    src/seed_matrix.cu
    src/depthmap_denoiser.cu
    src/reduction.cu
)

# Set sourcefiles
list(APPEND SOURCE_FILES
    src/depthmap.cpp
)

CUDA_ADD_LIBRARY(
    rpg_open_remode_cuda
    ${CU_SOURCE_FILES}
)

ADD_LIBRARY(
    rpg_open_remode
    ${SOURCE_FILES}
)

target_link_libraries(
    rpg_open_remode
    ${LINK_LIBS}
)

if(BUILD_ROS_NODE)
# Build the ROS node
add_executable(
  depthmap_node
  src/depthmap_node.cpp
  src/publisher.cpp
  src/main_ros.cpp
)
target_link_libraries(
  depthmap_node
  rpg_open_remode
  rpg_open_remode_cuda
)
# Build a node to publish the dataset
add_executable(
  dataset_publisher
  test/dataset.cpp
  test/publish_dataset.cpp
)
target_link_libraries(
  dataset_publisher
  ${LINK_LIBS}
)
endif(BUILD_ROS_NODE)

## Test on dataset
add_executable(
    dataset_main
    test/dataset_main.cpp
    test/dataset.cpp
)

target_link_libraries(
    dataset_main
    rpg_open_remode
    rpg_open_remode_cuda
)

# GTest
if(BUILD_TESTS)
enable_testing()
find_package(GTest REQUIRED)
# Accepts the following variable as input:
# GTEST_ROOT - (as a CMake or environment variable)
#              The root directory of the gtest install prefix

include_directories(${GTEST_ROOT}/include)
list(APPEND GTEST_LINK_LIBS
  ${GTEST_BOTH_LIBRARIES}
  pthread
)

CUDA_ADD_LIBRARY(
    rpg_open_remode_test
    test/copy.cu
    test/sobel.cu
)

# Set sourcefiles for the tests
list(APPEND TEST_SOURCE_FILES
  test/device_image_test.cpp
  test/seed_matrix_test.cpp
  test/main_test.cpp
  test/dataset.cpp
  test/epipolar_test.cpp
  test/reduction_test.cpp
)

add_executable(
  all_tests
  ${TEST_SOURCE_FILES}
)

target_link_libraries(
  all_tests
  rpg_open_remode
  rpg_open_remode_cuda
  rpg_open_remode_test
  ${GTEST_LINK_LIBS}
)

# Allow run of 'make test'
add_test(
  RMDCuTests
  all_tests --gtest_repeat=2
)
endif(BUILD_TESTS)
