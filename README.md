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

The following libraries are also required: `OpenCV`, `Eigen` and `Boost`.

#### Instructions

Install dependencies.

Ubuntu:
    
    sudo apt-get install libopencv-dev libeigen3-dev libboost-filesystem-dev

Clone the REMODE repository:

    git clone https://github.com/uzh-rpg/rpg_remode_legacy.git

Build the library and the tests:

    cd rpg_remode_legacy
    mkdir build && cd build
    cmake ..
    make -j7

Run the tests

    cd rpg_remode_legacy/build
    ./all_tests

Download the test dataset

    cd rpg_remode_legacy
    wget http://rpg.ifi.uzh.ch/datasets/traj_over_table_test_data.zip | unzip -xz

Run REMODE on the test data

   cd rpg_remode_legacy/build
   ./dataset_main
   
#### Contributing

You are very welcome to contribute to REMODE by opening a pull request via Github.
I try to follow the ROS C++ style guide http://wiki.ros.org/CppStyleGuide
