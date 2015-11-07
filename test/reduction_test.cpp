
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

TEST(RMDCuTests, deviceImageReduction)
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
  cv::Mat mask = h_in_img == TO_FIND;
  size_t cv_count = cv::countNonZero(mask);
  t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
  printf("Opencv execution time: %f seconds.\n", t);

  // CUDA execution
  size_t cu_count = rmd::countEqual(d_img, TO_FIND);

  std::cout << "DEBUG: OpenCV count=" << cv_count << ", CUDA count=" << cu_count << std::endl;
  ASSERT_EQ(cv_count, cu_count);
}
