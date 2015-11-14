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

#include <gtest/gtest.h>
#include <cuda_toolkit/helper_timer.h>
#include <opencv2/opencv.hpp>
#include <rmd/reduction.cuh>

// Test summing an image in device memory
TEST(deviceImageReduction, sum)
{
  const size_t w = 752;
  const size_t h = 480;

  cv::RNG rng;
  cv::Mat_<float> h_in_img(h, w);
  for(size_t r=0; r<h; ++r)
  {
    for(size_t c=0; c<w; ++c)
    {
      h_in_img.at<float>(r, c) = rng.uniform(0.0f, 1.0f);
    }
  }

  // Upload data to gpu memory
  rmd::DeviceImage<float> d_img(w, h);
  d_img.setDevData(reinterpret_cast<float*>(h_in_img.data));

  // OpenCV execution
  double t = (double)cv::getTickCount();
  d_img.getDevData(reinterpret_cast<float*>(h_in_img.data));
  cv::Scalar cv_sum = cv::sum(h_in_img);
  t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
  printf("Opencv execution time (including download of image from device memory): %f seconds.\n", t);

  // CUDA execution
  dim3 num_threads_per_block;
  dim3 num_blocks_per_grid;
  num_threads_per_block.x = 16;
  num_threads_per_block.y = 16;
  num_blocks_per_grid.x = 4;
  num_blocks_per_grid.y = 4;
  rmd::ImageReducer<float> img_reducer(num_threads_per_block, num_blocks_per_grid);

  StopWatchInterface * timer = NULL;
  sdkCreateTimer(&timer);
  sdkResetTimer(&timer);
  sdkStartTimer(&timer);
  float cu_sum = img_reducer.sum(d_img);
  sdkStopTimer(&timer);
  t = sdkGetAverageTimerValue(&timer) / 1000.0;
  printf("CUDA execution time (including download of result): %f seconds.\n", t);

  std::cout << "DEBUG: OpenCV sum=" << static_cast<float>(cv_sum.val[0]) << ", CUDA sum=" << cu_sum << std::endl;
  ASSERT_FLOAT_EQ(cv_sum.val[0], cu_sum);
}

// Test counting pixels in device memory
TEST(deviceImageReduction, countEqual)
{
  const size_t w = 752;
  const size_t h = 480;

  cv::RNG rng;
  cv::Mat_<int> h_in_img(h, w, 0);
  for(size_t r=0; r<h; ++r)
  {
    for(size_t c=0; c<w; ++c)
    {
      h_in_img.at<int>(r, c) = rng.uniform(0, 256);
    }
  }

  // Upload data to gpu memory
  rmd::DeviceImage<int> d_img(w, h);
  d_img.setDevData(reinterpret_cast<int*>(h_in_img.data));

  static const int TO_FIND = 2;

  // OpenCV execution
  double t = (double)cv::getTickCount();
  d_img.getDevData(reinterpret_cast<int*>(h_in_img.data));
  cv::Mat mask = h_in_img == TO_FIND;
  size_t cv_count = cv::countNonZero(mask);
  t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
  printf("Opencv execution time (including download of image from device memory): %f seconds.\n", t);

  // CUDA execution
  dim3 num_threads_per_block;
  dim3 num_blocks_per_grid;
  num_threads_per_block.x = 16;
  num_threads_per_block.y = 16;
  num_blocks_per_grid.x = 4;
  num_blocks_per_grid.y = 4;
  rmd::ImageReducer<int> img_reducer(num_threads_per_block, num_blocks_per_grid);

  StopWatchInterface * timer = NULL;
  sdkCreateTimer(&timer);
  sdkResetTimer(&timer);
  sdkStartTimer(&timer);
  size_t cu_count = img_reducer.countEqual(d_img, TO_FIND);
  sdkStopTimer(&timer);
  t = sdkGetAverageTimerValue(&timer) / 1000.0;
  printf("CUDA execution time (including download of result): %f seconds.\n", t);

  std::cout << "DEBUG: OpenCV count=" << cv_count << ", CUDA count=" << cu_count << std::endl;
  ASSERT_EQ(cv_count, cu_count);
}
