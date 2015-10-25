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

#include "../test/dataset.h"

#include "copy.cuh"
#include "sobel.cuh"

TEST(RMDCuTests, deviceImageUploadDownloadFloat)
{
  rmd::test::Dataset dataset;
  if(!dataset.loadPathFromEnv())
  {
    FAIL() << "could not retrieve dataset path from the environment variable '"
           << rmd::test::Dataset::getDataPathEnvVar();
  }
  cv::Mat img;
  if(!dataset.readImage(img, "scene_000.png"))
  {
    FAIL() << "could not could not load test image from dataset";
  }

  cv::Mat img_flt;
  img.convertTo(img_flt, CV_32F, 1./255.);

  const size_t w = img_flt.cols;
  const size_t h = img_flt.rows;
  // upload data to gpu memory
  rmd::DeviceImage<float> in_img(w, h);
  in_img.setDevData(reinterpret_cast<float*>(img_flt.data));

  float * cu_img = new float[w*h];
  in_img.getDevData(cu_img);

  for(size_t y=0; y<h; ++y)
  {
    for(size_t x=0; x<w; ++x)
    {
      ASSERT_FLOAT_EQ(img_flt.at<float>(y, x), cu_img[y*w+x]);
    }
  }
  delete cu_img;
}

TEST(RMDCuTests, deviceImageUploadDownloadFloat2)
{
  rmd::test::Dataset dataset;
  if(!dataset.loadPathFromEnv())
  {
    FAIL() << "could not retrieve dataset path from the environment variable '"
           << rmd::test::Dataset::getDataPathEnvVar();
  }
  cv::Mat img;
  if(!dataset.readImage(img, "scene_000.png"))
  {
    FAIL() << "could not could not load test image from dataset";
  }

  cv::Mat img_flt;
  img.convertTo(img_flt, CV_32F, 1./255.);

  // Opencv gradient computation
  cv::Mat ocv_grad_x, ocv_grad_y;
  cv::Sobel(img_flt, ocv_grad_x, CV_32F, 1, 0, CV_SCHARR);
  cv::Sobel(img_flt, ocv_grad_y, CV_32F, 0, 1, CV_SCHARR);

  const size_t w = img_flt.cols;
  const size_t h = img_flt.rows;

  float2 * cu_grad = new float2[w*h];
  for(size_t y=0; y<h; ++y)
  {
    for(size_t x=0; x<w; ++x)
    {
      cu_grad[y*w+x].x = ocv_grad_x.at<float>(y, x);
      cu_grad[y*w+x].y = ocv_grad_y.at<float>(y, x);
    }
  }

  // upload data to device memory
  rmd::DeviceImage<float2> in_img(w, h);
  in_img.setDevData(cu_grad);

  // download data to host memory
  memset(cu_grad, 0, sizeof(float2)*w*h);
  in_img.getDevData(cu_grad);

  for(size_t y=0; y<h; ++y)
  {
    for(size_t x=0; x<w; ++x)
    {
      ASSERT_FLOAT_EQ(ocv_grad_x.at<float>(y, x), cu_grad[y*w+x].x);
      ASSERT_FLOAT_EQ(ocv_grad_y.at<float>(y, x), cu_grad[y*w+x].y);
    }
  }
  delete cu_grad;
}

TEST(RMDCuTests, deviceImageCopyFloat)
{
  rmd::test::Dataset dataset;
  if(!dataset.loadPathFromEnv())
  {
    FAIL() << "could not retrieve dataset path from the environment variable '"
           << rmd::test::Dataset::getDataPathEnvVar();
  }
  cv::Mat img;
  if(!dataset.readImage(img, "scene_000.png"))
  {
    FAIL() << "could not could not load test image from dataset";
  }

  cv::Mat img_flt;
  img.convertTo(img_flt, CV_32F, 1./255.);

  const size_t w = img_flt.cols;
  const size_t h = img_flt.rows;

  // upload data to gpu memory
  rmd::DeviceImage<float> in_img(w, h);
  in_img.setDevData(reinterpret_cast<float*>(img_flt.data));

  // create copy on device
  rmd::DeviceImage<float> out_img(w, h);
  rmd::copy(in_img, out_img);

  float * cu_img = new float[w*h];
  out_img.getDevData(cu_img);

  for(size_t y=0; y<h; ++y)
  {
    for(size_t x=0; x<w; ++x)
    {
      ASSERT_FLOAT_EQ(img_flt.at<float>(y, x), cu_img[y*w+x]);
    }
  }
  delete cu_img;
}

TEST(RMDCuTests, deviceImageSobelTest)
{
  rmd::test::Dataset dataset;
  if(!dataset.loadPathFromEnv())
  {
    FAIL() << "could not retrieve dataset path from the environment variable '"
           << rmd::test::Dataset::getDataPathEnvVar();
  }
  cv::Mat img;
  if(!dataset.readImage(img, "scene_000.png"))
  {
    FAIL() << "could not could not load test image from dataset";
  }

  cv::Mat img_flt;
  img.convertTo(img_flt, CV_32F, 1./255.);

  // Compare results of the Scharr operator to compute image gradient
  const size_t w = img_flt.cols;
  const size_t h = img_flt.rows;

  // Opencv gradient computation
  cv::Mat ocv_grad_x(h, w, CV_32FC1);
  cv::Mat ocv_grad_y(h, w, CV_32FC1);
  double t = (double)cv::getTickCount();
  cv::Sobel(img_flt, ocv_grad_x, CV_32F, 1, 0, CV_SCHARR);
  cv::Sobel(img_flt, ocv_grad_y, CV_32F, 0, 1, CV_SCHARR);
  t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
  printf("Opencv execution time: %f seconds.\n", t);

  // CUDA gradient computation

  // upload data to device memory
  rmd::DeviceImage<float> in_img(w, h);
  in_img.setDevData(reinterpret_cast<float*>(img_flt.data));

  // compute gradient on device
  rmd::DeviceImage<float2> out_grad(w, h);

  StopWatchInterface * timer = NULL;
  sdkCreateTimer(&timer);
  sdkResetTimer(&timer);
  sdkStartTimer(&timer);
  rmd::sobel(in_img, out_grad);
  sdkStopTimer(&timer);
  t = sdkGetAverageTimerValue(&timer) / 1000.0;
  printf("CUDA execution time: %f seconds.\n", t);

  // download result to host memory
  float2 * cu_grad = new float2[w*h];
  out_grad.getDevData(cu_grad);

  for(size_t y=1; y<h-1; ++y)
  {
    for(size_t x=1; x<w-1; ++x)
    {
      ASSERT_NEAR(ocv_grad_x.at<float>(y, x), cu_grad[y*w+x].x, 0.00001f);
      ASSERT_NEAR(ocv_grad_y.at<float>(y, x), cu_grad[y*w+x].y, 0.00001f);
    }
  }
  delete cu_grad;
}


TEST(RMDCuTests, deviceImageSobelTexTest)
{
  rmd::test::Dataset dataset;
  if(!dataset.loadPathFromEnv())
  {
    FAIL() << "could not retrieve dataset path from the environment variable '"
           << rmd::test::Dataset::getDataPathEnvVar();
  }
  cv::Mat img;
  if(!dataset.readImage(img, "scene_000.png"))
  {
    FAIL() << "could not could not load test image from dataset";
  }

  cv::Mat img_flt;
  img.convertTo(img_flt, CV_32F, 1./255.);

  // Compare results of the Scharr operator to compute image gradient
  const size_t w = img_flt.cols;
  const size_t h = img_flt.rows;

  // Opencv gradient computation
  cv::Mat ocv_grad_x(h, w, CV_32FC1);
  cv::Mat ocv_grad_y(h, w, CV_32FC1);
  double t = (double)cv::getTickCount();
  cv::Sobel(img_flt, ocv_grad_x, CV_32F, 1, 0, CV_SCHARR);
  cv::Sobel(img_flt, ocv_grad_y, CV_32F, 0, 1, CV_SCHARR);
  t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
  printf("Opencv execution time: %f seconds.\n", t);

  // CUDA gradient computation

  // upload data to device memory
  rmd::DeviceImage<float> in_img(w, h);
  in_img.setDevData(reinterpret_cast<float*>(img_flt.data));

  // compute gradient on device
  rmd::DeviceImage<float2> out_grad(w, h);

  StopWatchInterface * timer = NULL;
  sdkCreateTimer(&timer);
  sdkResetTimer(&timer);
  sdkStartTimer(&timer);
  rmd::sobelTex(in_img, out_grad);
  sdkStopTimer(&timer);
  t = sdkGetAverageTimerValue(&timer) / 1000.0;
  printf("CUDA execution time: %f seconds.\n", t);

  // download result to host memory
  float2 * cu_grad = new float2[w*h];
  out_grad.getDevData(cu_grad);

  for(size_t y=1; y<h-1; ++y)
  {
    for(size_t x=1; x<w-1; ++x)
    {
      ASSERT_NEAR(ocv_grad_x.at<float>(y, x), cu_grad[y*w+x].x, 0.00001f);
      ASSERT_NEAR(ocv_grad_y.at<float>(y, x), cu_grad[y*w+x].y, 0.00001f);
    }
  }
  delete cu_grad;
}
