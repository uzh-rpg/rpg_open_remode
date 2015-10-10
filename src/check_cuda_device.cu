// This file is part of REMODE - REgularized MOnocular Depth Estimation.
//
// Copyright (C) 2014 Matia Pizzoli <matia dot pizzoli at gmail dot com>
// Robotics and Perception Group, University of Zurich, Switzerland
// http://rpg.ifi.uzh.ch
//
// REMODE is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// REMODE is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <cstdio>
#include <string>
#include <cuda_runtime.h>
#include <rmd/check_cuda_device.cuh>

bool rmd::checkCudaDevice(int argc, char **argv)
{
  cudaDeviceProp deviceProp;
  deviceProp.major = 0;
  deviceProp.minor = 0;

  int devID = 0;
  const std::string device_arg("--device=");
  for(int i=1; i<argc; ++i)
  {
    const std::string arg(argv[i]);
    if(device_arg == arg.substr(0, device_arg.size()))
    {
      devID = atoi(arg.substr(device_arg.size(), arg.size()).c_str());
      printf("User-specified device: %d\n", devID);
      break;
    }
  }

  cudaError err = cudaSetDevice(devID);
  if(cudaSuccess != err)
  {
    printf("ERROR: cudaSetDevice %s\n", cudaGetErrorString(err));
    return false;
  }
  err = cudaGetDeviceProperties(&deviceProp, devID);
  if(cudaSuccess != err)
  {
    printf("ERROR: cudaGetDeviceProperties %s\n", cudaGetErrorString(err));
    return false;
  }
  printf("Using GPU device %d: \"%s\" with compute capability %d.%d\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
  printf("GPU device %d has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
         devID, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

  const int version = (deviceProp.major * 0x10 + deviceProp.minor);

  if (version < 0x20)
  {
    printf("ERROR: a minimum CUDA compute 2.0 capability is required.\n");
    return false;
  }

  return true;
}
