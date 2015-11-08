REMODE
===

This repository contains an implementation of REMODE (REgularized MOnocular Depth Estimation), as described in the paper

http://rpg.ifi.uzh.ch/docs/ICRA14_Pizzoli.pdf

The following video demonstrates the proposed approach:

http://youtu.be/QTKd5UWCG0Q

#### Disclaimer

The REMODE implementation in this repository is research code, any fitness for a particular purpose is disclaimed.

#### Licence

The source code is released under a GPLv3 licence.

http://www.gnu.org/licenses/

#### Citing

If you use REMODE in an academic context, please cite the following publication:

    @inproceedings{Pizzoli2014ICRA,
      author = {Pizzoli, Matia and Forster, Christian and Scaramuzza, Davide},
      title = {{REMODE}: Probabilistic, Monocular Dense Reconstruction in Real Time},
      booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
      year = {2014}
    }

#### Requirements

This implementation requires a CUDA capable GPU and the NVIDIA CUDA Toolkit

https://developer.nvidia.com/cuda-zone

The following libraries are also required: `OpenCV`, `Eigen`, `Boost` and `googletest`

#### Instructions

Install dependencies. The following assumes you have an environment variable `MY_WORKSPACE` pointing to your workspace.

Ubuntu:
    
    sudo apt-get install libopencv-dev libeigen3-dev libboost-filesystem-dev
    
Install googletest 1.7. Several Ubuntu packages install googletest as a dependency. In order to avoid conflicts with different libray versions, build googletest in `$MY_WORKSPACE`:
    
    cd $MY_WORKSPACE
    git clone https://github.com/google/googletest.git
    cd googletest
    git checkout release-1.7.0
    mkdir build && cd build
    cmake ..
    make
    cd ..
    mkdir install
    cp -r include install
    cp build/*.a install
    
#### Build without ROS

Clone the REMODE repository:

    cd $MY_WORKSPACE
    git clone https://github.com/uzh-rpg/rpg_open_remode.git

Build the library and the tests:

    cd $MY_WORKSPACE/rpg_open_remode
    mkdir build && cd build
    cmake -DGTEST_ROOT=$MY_WORKSPACE/googletest/install -DBUILD_ROS_NODE=OFF ..
    make
    
#### Build the ROS node (using catkin)

Assuming that the environment variable `MY_CATKIN_WORKSPACE` points to your Catkin workspace, clone the repository:

    cd $MY_CATKIN_WORKSPACE/src
    git clone https://github.com/uzh-rpg/rpg_open_remode.git
    
Build the library, the tests and the ROS node:

    cd ..
    source devel/setup.sh
    catkin_make -DGTEST_ROOT=$MY_WORKSPACE/googletest/install
    
#### Test

Download the test dataset

    cd $MY_WORKSPACE/rpg_open_remode
    wget http://rpg.ifi.uzh.ch/datasets/remode_test_data.zip
    unzip remode_test_data.zip

Specify the path to the test data by setting the environment variable

    export RMD_TEST_DATA_PATH=$MY_WORKSPACE/rpg_open_remode/test_data

If you built without ROS, just execute

    cd $MY_WORKSPACE/rpg_open_remode/build
    ./all_tests
    
Otherwise

    cd $MY_CATKIN_WORKSPACE/devel/lib/open_remode
    ./all_tests

You can now run REMODE on the test data. Specify the path to the test data:

    export RMD_TEST_DATA_PATH=$MY_WORKSPACE/rpg_open_remode/test_data

In case you did not build the ROS node, just execute the following:

    cd $MY_WORKSPACE/rpg_open_remode/build
    ./dataset_main
    
For the ROS build, a launch file is provided to start the REMODE ROS node. Update your ROS package path and launch the REMODE node:

    source cd $MY_CATKIN_WORKSPACE/devel/setup.sh
    roslaunch open_remode dataset.launch
    
A node is provided to publish the test dataset. Run it by executing

    rosrun open_remode dataset_publisher
    
#### CUDA specific topics
    
If you have more than one CUDA-capable GPUs in your system, you can specify which one to use by passing the `--device=` command-line argument. For instance, to run the example using the GPU identified by ID 1, execute the following:

    cd $MY_WORKSPACE/rpg_open_remode/build
    ./dataset_main --device=1

By default, the GPU identified by ID 0 is used.

You can target the compute capability of your CUDA device by specifying `arch=compute_xx,code=sm_x` in the `CUDA_NVCC_FLAGS`.
  
#### Acknowledgments

The author acknowledges the key contributions by Christian Forster and Manuel Werlberger. 
   
#### Contributing

You are very welcome to contribute to REMODE by opening a pull request via Github.
I try to follow the ROS C++ style guide http://wiki.ros.org/CppStyleGuide
