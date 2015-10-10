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
  printf("Running executable: %s\nChecking available CUDA-capable devices...\n", argv[0]);

  int dev_cnt;
  cudaError err = cudaGetDeviceCount(&dev_cnt);
  if(cudaSuccess != err)
  {
    printf("ERROR: cudaGetDeviceCount %s\n", cudaGetErrorString(err));
    return false;
  }

  if(0 == dev_cnt)
  {
    printf("ERROR: no CUDA-capable device found.\n");
    return false;
  }

  cudaDeviceProp device_prop;
  device_prop.major = 0;
  device_prop.minor = 0;

  printf("%d CUDA-capable GPU detected:\n", dev_cnt);
  int dev_id;
  for(dev_id=0; dev_id<dev_cnt; ++dev_id)
  {
    err = cudaGetDeviceProperties(&device_prop, dev_id);
    if(cudaSuccess != err)
    {
      printf("ERROR: cudaGetDeviceProperties could not get properties for device %d. %s\n", dev_id, cudaGetErrorString(err));
    }
    else
    {
      printf("Device %d - %s\n", dev_id, device_prop.name);
    }
  }

  dev_id = 0;
  const std::string device_arg("--device=");
  for(int i=1; i<argc; ++i)
  {
    const std::string arg(argv[i]);
    if(device_arg == arg.substr(0, device_arg.size()))
    {
      dev_id = atoi(arg.substr(device_arg.size(), arg.size()).c_str());
      printf("User-specified device: %d\n", dev_id);
      break;
    }
  }

  if( (dev_id<0) || (dev_id>dev_cnt-1) )
  {
    printf("ERROR: invalid device ID specified. Please specify a value in [0, %d].\n", dev_cnt-1);
    return false;
  }

  err = cudaGetDeviceProperties(&device_prop, dev_id);
  if(cudaSuccess != err)
  {
    printf("ERROR: cudaGetDeviceProperties %s\n", cudaGetErrorString(err));
    return false;
  }
  printf("Using GPU device %d: \"%s\" with compute capability %d.%d\n", dev_id, device_prop.name, device_prop.major, device_prop.minor);
  printf("GPU device %d has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
         dev_id, device_prop.multiProcessorCount, device_prop.major, device_prop.minor);

  const int version = (device_prop.major * 0x10 + device_prop.minor);

  if (version < 0x20)
  {
    printf("ERROR: a minimum CUDA compute 2.0 capability is required.\n");
    return false;
  }

  if (cudaComputeModeProhibited == device_prop.computeMode)
  {
    printf("ERROR: device is running in 'Compute Mode Prohibited'\n");
    return false;
  }

  if (device_prop.major < 1)
  {
    printf("ERROR: device %d is not a CUDA-capable GPU.\n", dev_id);
    return false;
  }

  err = cudaSetDevice(dev_id);
  if(cudaSuccess != err)
  {
    printf("ERROR: cudaSetDevice %s\n", cudaGetErrorString(err));
    return false;
  }

  return true;
}
