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

SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_30,code=sm_30; --ptxas-options=-v) #35 on supercmoputer!
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
find_package(CUDA REQUIRED)

## System dependencies are found with CMake's conventions
find_package(OpenCV REQUIRED)
find_package(Eigen REQUIRED)
find_package(Boost REQUIRED)
# find_package(Sophus REQUIRED)
# find_package(nanoflann REQUIRED)

## Specify additional locations of header files
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
#  ${Sophus_INCLUDE_DIRS}
#  ${nanoflann_INCLUDE_DIRS}

)

# Depth Map lib
# Maximum extent of epipolar line search in pixels
ADD_DEFINITIONS(-DMAX_EXTENT_EPIPOLAR_SEARCH=100)

CUDA_ADD_LIBRARY(
    remode
    src/padded_memory.cu
    src/seed_matrix.cu
#    src/tvl1.cpp
#    src/depth_map.cu
#    src/depthmap_denoiser.cu
#    src/remode_base.cpp
#    src/remode_gpu.cpp
#    src/result_publisher.cpp
)


#TARGET_LINK_LIBRARIES(remode
#  ${catkin_LIBRARIES}
#  ${OpenCV_LIBS}
#  ${Sophus_LIBRARIES}
#)

## Declare a cpp executable
add_executable(
    dataset_main
    src/depthmap.cpp
    src/test/dataset_main.cpp
)
#add_executable(test_cu_depth_map test/test_cu_depth_map.cpp)
#add_executable(test_cu_matcher test/test_cu_matcher_main.cpp)
#add_executable(plot_epipolar_line test/plot_epipolar_line.cpp)
#add_executable(remode_live src/node_live.cpp)
#add_executable(remode_benchmark src/node_benchmark.cpp)
#add_executable(remode_test_tvl1 test/test_tvl1.cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(
    dataset_main
    remode
    boost_system
    boost_filesystem
    ${OpenCV_LIBS}
)
#target_link_libraries(remode_benchmark remode)
#target_link_libraries(remode_test_tvl1 remode)
