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

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>

#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

#include <rmd/depthmap.h>
#include <rmd/check_cuda_device.cuh>

#include "../test/dataset.h"

int main(int argc, char **argv)
{
  if(!rmd::checkCudaDevice(argc, argv))
    return EXIT_FAILURE;

  rmd::PinholeCamera cam(481.2f, -480.0f, 319.5f, 239.5f);

  rmd::test::Dataset dataset("first_200_frames_traj_over_table_input_sequence.txt");
  if(!dataset.loadPathFromEnv())
  {
    std::cerr << "ERROR: could not retrieve dataset path from the environment variable '"
              << rmd::test::Dataset::getDataPathEnvVar() <<"'" << std::endl;
  }
  if (!dataset.readDataSequence(0, 200))
  {
    std::cerr << "ERROR: could not read dataset" << std::endl;
    return EXIT_FAILURE;
  }

  const size_t width  = 640;
  const size_t height = 480;

  bool first_img = true;
  rmd::Depthmap depthmap(width, height, cam.fx, cam.cx, cam.fy, cam.cy);

  // store the timings
  // update
  std::vector<double> update_time;

  for(const auto data : dataset)
  {
    cv::Mat img;
    if(!dataset.readImage(img, data))
    {
      std::cerr << "ERROR: could not read image " << data.getImageFileName() << std::endl;
      continue;
    }

    cv::Mat depth_32FC1;
    if(!dataset.readDepthmap(depth_32FC1, data, img.cols, img.rows))
    {
      std::cerr << "ERROR: could not read depthmap " << data.getDepthmapFileName() << std::endl;
      continue;
    }
    double min_depth, max_depth;
    cv::minMaxLoc(depth_32FC1, &min_depth, &max_depth);

    rmd::SE3<float> T_world_curr;
    dataset.readCameraPose(T_world_curr, data);

    std::cout << "RUN EXPERIMENT: inputting image " << data.getImageFileName() <<  std::endl;
    std::cout << "T_world_curr:" << std::endl;
    std::cout << T_world_curr << std::endl;

    // process
    if(first_img)
    {
      if(depthmap.setReferenceImage(img, T_world_curr.inv(), min_depth, max_depth))
      {
        first_img = false;
      }
      else
      {
        std::cerr << "ERROR: could not set reference image" << std::endl;
        return EXIT_FAILURE;
      }
    }
    else
    {
      double t = (double)cv::getTickCount();
      depthmap.update(img, T_world_curr.inv());
      t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
      printf("\nUPDATE execution time: %f seconds.\n", t);
      update_time.push_back(t);
    }
  }

  // show depthmap
  depthmap.downloadDepthmap();
  cv::Mat result = depthmap.getDepthmap();
  cv::Mat colored = rmd::Depthmap::scaleMat(result);
  cv::imshow("result", colored);

  // denoise
  depthmap.downloadDenoisedDepthmap(0.5f, 200);
  cv::Mat denoised_result = depthmap.getDepthmap();
  cv::Mat colored_denoised = rmd::Depthmap::scaleMat(denoised_result);
  cv::imshow("denoised_result", colored_denoised);

  cv::waitKey();

  // time statistics
  const double time_mean = std::accumulate(update_time.begin(), update_time.end(), 0.0) / static_cast<double>(update_time.size());
  double time_var = 0.0;
  for(const auto & t : update_time)
  {
    time_var += (t-time_mean)*(t-time_mean);
  }
  time_var /= static_cast<double>(update_time.size());

  std::cout << "\n\n";
  std::cout << "MEAN update time: " << time_mean << std::endl;
  std::cout << "VAR  update time: " << time_var  << std::endl
            << "(STDDEV: " << std::sqrt(time_var) << ")" << std::endl;

  return EXIT_SUCCESS;
}
