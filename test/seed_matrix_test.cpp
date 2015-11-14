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

#include <rmd/seed_matrix.cuh>
#include <rmd/se3.cuh>

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

#include "dataset.h"

TEST(RMDCuTests, seedMatrixInit)
{
  rmd::PinholeCamera cam(481.2f, -480.0f, 319.5f, 239.5f);

  rmd::test::Dataset dataset("first_200_frames_traj_over_table_input_sequence.txt");
  if(!dataset.loadPathFromEnv())
  {
    std::cerr << "ERROR: could not retrieve dataset path from the environment variable '"
              << rmd::test::Dataset::getDataPathEnvVar() <<"'" << std::endl;
  }

  if (!dataset.readDataSequence())
    FAIL() << "could not read dataset";

  const size_t ref_ind = 1;
  const size_t curr_ind = 20;

  const auto ref_entry = dataset(ref_ind);
  cv::Mat ref_img;
  dataset.readImage(ref_img, ref_entry);
  cv::Mat ref_img_flt;
  ref_img.convertTo(ref_img_flt, CV_32F, 1.0f/255.0f);

  cv::Mat ref_depthmap;
  dataset.readDepthmap(ref_depthmap, ref_entry, ref_img.cols, ref_img.rows);

  rmd::SE3<float> T_world_ref;
  dataset.readCameraPose(T_world_ref, ref_entry);

  const auto curr_entry = dataset(curr_ind);
  cv::Mat curr_img;
  dataset.readImage(curr_img, curr_entry);
  cv::Mat curr_img_flt;
  curr_img.convertTo(curr_img_flt, CV_32F, 1.0f/255.0f);

  rmd::SE3<float> T_world_curr;
  dataset.readCameraPose(T_world_curr, curr_entry);

  const float min_scene_depth = 0.4f;
  const float max_scene_depth = 1.8f;

  rmd::SeedMatrix seeds(ref_img.cols, ref_img.rows, cam);

  StopWatchInterface * timer = NULL;
  sdkCreateTimer(&timer);
  sdkResetTimer(&timer);
  sdkStartTimer(&timer);

  seeds.setReferenceImage(
        reinterpret_cast<float*>(ref_img_flt.data),
        T_world_ref.inv(),
        min_scene_depth,
        max_scene_depth);

  sdkStopTimer(&timer);
  double t = sdkGetAverageTimerValue(&timer) / 1000.0;
  printf("setReference image CUDA execution time: %f seconds.\n", t);

  cv::Mat initial_depthmap(ref_img.rows, ref_img.cols, CV_32FC1);
  seeds.downloadDepthmap(reinterpret_cast<float*>(initial_depthmap.data));

  cv::Mat initial_sigma_sq(ref_img.rows, ref_img.cols, CV_32FC1);
  seeds.downloadSigmaSq(reinterpret_cast<float*>(initial_sigma_sq.data));

  cv::Mat initial_a(ref_img.rows, ref_img.cols, CV_32FC1);
  seeds.downloadA(reinterpret_cast<float*>(initial_a.data));

  cv::Mat initial_b(ref_img.rows, ref_img.cols, CV_32FC1);
  seeds.downloadB(reinterpret_cast<float*>(initial_b.data));

  const float avg_scene_depth = (min_scene_depth+max_scene_depth)/2.0f;
  const float max_scene_sigma_sq = (max_scene_depth - min_scene_depth) * (max_scene_depth - min_scene_depth) / 36.0f;
  for(size_t r=0; r<ref_img.rows; ++r)
  {
    for(size_t c=0; c<ref_img.cols; ++c)
    {
      ASSERT_FLOAT_EQ(avg_scene_depth, initial_depthmap.at<float>(r, c));
      ASSERT_FLOAT_EQ(max_scene_sigma_sq, initial_sigma_sq.at<float>(r, c));
      ASSERT_FLOAT_EQ(10.0f, initial_a.at<float>(r, c));
      ASSERT_FLOAT_EQ(10.0f, initial_b.at<float>(r, c));
    }
  }

  // Test initialization of NCC template statistics

  // CUDA computation
  cv::Mat cu_sum_templ(ref_img.rows, ref_img.cols, CV_32FC1);
  seeds.downloadSumTempl(reinterpret_cast<float*>(cu_sum_templ.data));
  cv::Mat cu_const_templ_denom(ref_img.rows, ref_img.cols, CV_32FC1);
  seeds.downloadConstTemplDenom(reinterpret_cast<float*>(cu_const_templ_denom.data));

  // Host computation
  cv::Mat ocv_sum_templ(ref_img.rows, ref_img.cols, CV_32FC1);
  cv::Mat ocv_const_templ_denom(ref_img.rows, ref_img.cols, CV_32FC1);

  for(size_t y=RMD_CORR_PATCH_SIDE; y<ref_img.rows-RMD_CORR_PATCH_SIDE/2; ++y)
  {
    for(size_t x=RMD_CORR_PATCH_SIDE; x<ref_img.cols-RMD_CORR_PATCH_SIDE/2; ++x)
    {
      double sum_templ    = 0.0f;
      double sum_templ_sq = 0.0f;
      for(int patch_y=0; patch_y<RMD_CORR_PATCH_SIDE; ++patch_y)
      {
        for(int patch_x=0; patch_x<RMD_CORR_PATCH_SIDE; ++patch_x)
        {
          const double templ = (double) ref_img_flt.at<float>( y-RMD_CORR_PATCH_SIDE/2+patch_y, x-RMD_CORR_PATCH_SIDE/2+patch_x );
          sum_templ += templ;
          sum_templ_sq += templ*templ;
        }
      }
      ocv_sum_templ.at<float>(y, x) = (float) sum_templ;
      ocv_const_templ_denom.at<float>(y, x) = (float) ( ((double)RMD_CORR_PATCH_AREA)*sum_templ_sq - sum_templ*sum_templ );
    }
  }
  for(size_t r=RMD_CORR_PATCH_SIDE; r<ref_img.rows-RMD_CORR_PATCH_SIDE/2; ++r)
  {
    for(size_t c=RMD_CORR_PATCH_SIDE; c<ref_img.cols-RMD_CORR_PATCH_SIDE/2; ++c)
    {
      ASSERT_NEAR(ocv_sum_templ.at<float>(r, c), cu_sum_templ.at<float>(r, c), 0.00001f);
      ASSERT_NEAR(ocv_const_templ_denom.at<float>(r, c), cu_const_templ_denom.at<float>(r, c), 0.001f);
    }
  }

}

TEST(RMDCuTests, seedMatrixCheck)
{
  rmd::PinholeCamera cam(481.2f, -480.0f, 319.5f, 239.5f);

  rmd::test::Dataset dataset("first_200_frames_traj_over_table_input_sequence.txt");
  if(!dataset.loadPathFromEnv())
  {
    std::cerr << "ERROR: could not retrieve dataset path from the environment variable '"
              << rmd::test::Dataset::getDataPathEnvVar() <<"'" << std::endl;
  }

  if (!dataset.readDataSequence())
    FAIL() << "could not read dataset";

  const size_t ref_ind = 1;
  const size_t curr_ind = 20;

  const auto ref_entry = dataset(ref_ind);
  cv::Mat ref_img;
  dataset.readImage(ref_img, ref_entry);
  cv::Mat ref_img_flt;
  ref_img.convertTo(ref_img_flt, CV_32F, 1.0f/255.0f);

  cv::Mat ref_depthmap;
  dataset.readDepthmap(ref_depthmap, ref_entry, ref_img.cols, ref_img.rows);

  rmd::SE3<float> T_world_ref;
  dataset.readCameraPose(T_world_ref, ref_entry);

  const auto curr_entry = dataset(curr_ind);
  cv::Mat curr_img;
  dataset.readImage(curr_img, curr_entry);
  cv::Mat curr_img_flt;
  curr_img.convertTo(curr_img_flt, CV_32F, 1.0f/255.0f);

  rmd::SE3<float> T_world_curr;
  dataset.readCameraPose(T_world_curr, curr_entry);

  const float min_scene_depth = 0.4f;
  const float max_scene_depth = 1.8f;

  rmd::SeedMatrix seeds(ref_img.cols, ref_img.rows, cam);

  seeds.setReferenceImage(
        reinterpret_cast<float*>(ref_img_flt.data),
        T_world_ref.inv(),
        min_scene_depth,
        max_scene_depth);

  StopWatchInterface * timer = NULL;
  sdkCreateTimer(&timer);
  sdkResetTimer(&timer);
  sdkStartTimer(&timer);

  seeds.update(
        reinterpret_cast<float*>(ref_img_flt.data),
        T_world_curr.inv());

  sdkStopTimer(&timer);
  double t = sdkGetAverageTimerValue(&timer) / 1000.0;
  printf("update CUDA execution time: %f seconds.\n", t);

  cv::Mat cu_convergence(ref_img.rows, ref_img.cols, CV_32SC1);
  seeds.downloadConvergence(reinterpret_cast<int*>(cu_convergence.data));

  for(size_t r=0; r<ref_img.rows; ++r)
  {
    for(size_t c=0; c<ref_img.cols; ++c)
    {
      if(r>ref_img.rows-RMD_CORR_PATCH_SIDE-1
         || r<RMD_CORR_PATCH_SIDE
         || c>ref_img.cols-RMD_CORR_PATCH_SIDE-1
         || c<RMD_CORR_PATCH_SIDE)
      {
        ASSERT_EQ(rmd::ConvergenceStates::BORDER, cu_convergence.at<int>(r, c)) << "(r, c) = (" << r << ", " << c <<")";
      }
      else
      {
        const int result = cu_convergence.at<int>(r, c);
        const bool success = (result == rmd::ConvergenceStates::UPDATE      ||
                              result == rmd::ConvergenceStates::DIVERGED    ||
                              result == rmd::ConvergenceStates::CONVERGED   ||
                              result == rmd::ConvergenceStates::NOT_VISIBLE ||
                              result == rmd::ConvergenceStates::NO_MATCH    );
        ASSERT_EQ(true, success) << "(r, c) = (" << r << ", " << c <<")";
      }
    }
  }

}
