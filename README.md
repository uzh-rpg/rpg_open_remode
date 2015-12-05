REMODE
===

This repository contains an implementation of REMODE (REgularized MOnocular Depth Estimation), as described in the paper

http://rpg.ifi.uzh.ch/docs/ICRA14_Pizzoli.pdf

The following video demonstrates the proposed approach:

http://youtu.be/QTKd5UWCG0Q

#### Disclaimer

The REMODE implementation in this repository is research code, any fitness for a particular purpose is disclaimed.

The code has been tested in Ubuntu 12.04, 14.04, 15.04, ROS Groovy, ROS Indigo and ROS Jade.

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
    
#### Install and run REMODE

The wiki 

https://github.com/uzh-rpg/rpg_open_remode/wiki

contains instructions on how to build and run REMODE.

> __NOTE__: this implementation requires a CUDA capable GPU and the NVIDIA CUDA Toolkit

https://developer.nvidia.com/cuda-zone
  
#### Acknowledgments

The author acknowledges the key contributions by Christian Forster, Manuel Werlberger and Jeff Delmerico.

Also, thanks to Michael Gassner, Zichao Zhang and Henri Rebecq for their valuable comments and help.
   
#### Contributing

You are very welcome to contribute to REMODE by opening a pull request via Github.
I try to follow the ROS C++ style guide http://wiki.ros.org/CppStyleGuide
