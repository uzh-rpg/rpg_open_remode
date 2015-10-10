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
    
Install googletest 1.7. Several Ubuntu packages install googletest as a dependency. In order to avoid conflicts with different libray versions, build googletest in $MY_WORKSPACE:
    
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

Clone the REMODE repository:

    cd $MY_WORKSPACE
    git clone https://github.com/uzh-rpg/rpg_open_remode.git

Build the library and the tests:

    cd $MY_WORKSPACE/rpg_open_remode
    mkdir build && cd build
    cmake -DGTEST_ROOT=$MY_WORKSPACE/googletest/install ..
    make
    
Download the test dataset

    cd $MY_WORKSPACE/rpg_open_remode
    wget http://rpg.ifi.uzh.ch/datasets/remode_test_data.zip
    unzip remode_test_data.zip

Run the tests

    cd $MY_WORKSPACE/rpg_open_remode/build
    ./all_tests

Run REMODE on the test data

    cd $MY_WORKSPACE/rpg_open_remode/build
    ./dataset_main
    
If you have more than one CUDA-capable GPUs in your system, you can specify which one to use by passing the `--device=` command-line argument. For instance, to run the example using the GPU identified with ID 1, execute the following:

    cd $MY_WORKSPACE/rpg_open_remode/build
    ./dataset_main --device=1

By default, the GPU identified by ID 0 is used.
  
#### Acknowledgments

The author acknowledges the key contributions by Christian Forster and Manuel Werlberger. 
   
#### Contributing

You are very welcome to contribute to REMODE by opening a pull request via Github.
I try to follow the ROS C++ style guide http://wiki.ros.org/CppStyleGuide
