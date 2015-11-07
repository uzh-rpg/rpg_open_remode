
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

  cv::Mat_<int> h_in_img(h, w, 0);
  h_in_img.at<int>(100, 100) = 1;
//  h_in_img.at<int>(200, 1) = 2;
//  h_in_img.at<int>(1, 1) = 4;
  h_in_img.at<int>(h-2, 1) = 255;
//  h_in_img.at<int>(1, w-2) = 2;
//  h_in_img.at<int>(h-2, w-2) = 4;

  // upload data to gpu memory
  rmd::DeviceImage<int> d_img(w, h);
  d_img.setDevData(reinterpret_cast<int*>(h_in_img.data));

  size_t count = rmd::countEqual(d_img, 255);
  std::cout << "COUNT = " << count << std::endl;
/*
  cv::Mat_<int> h_out_img(h, w);
  d_img.getDevData(reinterpret_cast<int*>(h_out_img.data));

  cv::Mat img_8uc1;
  h_out_img.convertTo(img_8uc1, CV_8UC1);

  cv::imshow("host_out_img", img_8uc1);
  cv::waitKey();
  */
}
