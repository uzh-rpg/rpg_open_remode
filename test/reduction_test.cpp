
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

  static const int TO_FIND = 2;

  cv::Mat_<int> h_in_img(h, w, 0);
  h_in_img.at<int>(1, w-2) = 2;
  h_in_img.at<int>(h-2, w-2) = 4;
  h_in_img.at<int>(100, 100) = 1;
  h_in_img.at<int>(200, 200) = 2;
  h_in_img.at<int>(300, 300) = 3;
  h_in_img.at<int>(400, 400) = 4;
  h_in_img.at<int>(150, 100)  = 4;
  h_in_img.at<int>(275, 430) = 4;

  // OpenCV execution
  double t = (double)cv::getTickCount();
  cv::Mat mask = h_in_img == TO_FIND;
  size_t cv_count = cv::countNonZero(mask);
  t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
  printf("Opencv execution time: %f seconds.\n", t);

  std::cout << "CV COUNT: " << cv_count << std::endl;

  // upload data to gpu memory
  rmd::DeviceImage<int> d_img(w, h);
  d_img.setDevData(reinterpret_cast<int*>(h_in_img.data));

  // CUDA execution
  int cu_sum = rmd::sum(d_img);
  std::cout << "CUDA SUM: " << cu_sum << std::endl;

  StopWatchInterface * timer = NULL;
  sdkCreateTimer(&timer);
  sdkResetTimer(&timer);
  sdkStartTimer(&timer);
  size_t cu_count = rmd::countEqual(d_img, TO_FIND);
  sdkStopTimer(&timer);
  t = sdkGetAverageTimerValue(&timer) / 1000.0;
  printf("CUDA execution time: %f seconds.\n", t);
  std::cout << "CUDA COUNT: " << cu_count << std::endl;

/*
  cv::Mat_<int> h_out_img(h, w);
  d_img.getDevData(reinterpret_cast<int*>(h_out_img.data));

  cv::Mat img_8uc1;
  h_out_img.convertTo(img_8uc1, CV_8UC1);

  cv::imshow("host_out_img", img_8uc1);
  cv::waitKey();
  */
}
