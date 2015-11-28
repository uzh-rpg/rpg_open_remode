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
#include <rmd/helper_vector_types.cuh>

#include <rmd/seed_matrix.cuh>
#include <rmd/depthmap.h>
#include <rmd/se3.cuh>

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

#include "dataset.h"

struct CallbackData
{
  cv::Mat *ref_img;
  cv::Mat *curr_img;
  cv::Mat *ref_depthmap;
  rmd::PinholeCamera *cam;
  rmd::SE3<float> *T_curr_ref;
};

void mouseCallback(int event, int x, int y, int flags, void *userdata)
{
  const CallbackData &cb_data = *static_cast<CallbackData*>(userdata);
  if(event == cv::EVENT_LBUTTONDOWN)
  {
    const float  depth   = cb_data.ref_depthmap->at<float>(y, x);
    const float2 ref_px  = make_float2((float)x, (float)y);
    const float3 ref_f   = normalize(cb_data.cam->cam2world(ref_px));
    const float2 curr_px = cb_data.cam->world2cam( *cb_data.T_curr_ref * (ref_f * depth ) );
    printf("(%d, %d), d = %f -> (%f, %f)\n", x, y, depth, curr_px.x, curr_px.y);
    cv::circle(*cb_data.ref_img, cv::Point(x, y), 3, cv::Scalar(255));
    cv::imshow("ref", *cb_data.ref_img);
    cv::circle(*cb_data.curr_img, cv::Point(curr_px.x, curr_px.y), 3, cv::Scalar(255));

    Eigen::Matrix3f K;
    // Camera intrinsics
    K << cb_data.cam->fx, 0.0f, cb_data.cam->cx,
        0.0f, cb_data.cam->fy, cb_data.cam->cy,
        0.0f, 0.0f, 1.0f;
    // R_curr_ref
    Eigen::Matrix3f R;
    R << cb_data.T_curr_ref->data(0, 0), cb_data.T_curr_ref->data(0, 1), cb_data.T_curr_ref->data(0, 2),
         cb_data.T_curr_ref->data(1, 0), cb_data.T_curr_ref->data(1, 1), cb_data.T_curr_ref->data(1, 2),
         cb_data.T_curr_ref->data(2, 0), cb_data.T_curr_ref->data(2, 1), cb_data.T_curr_ref->data(2, 2);
    // t_x is the Skew-simmetric matrix of t
    Eigen::Matrix3f t_x;
    t_x << 0.0f, -cb_data.T_curr_ref->data(2, 3), cb_data.T_curr_ref->data(1, 3),
        cb_data.T_curr_ref->data(2, 3), 0.0f, -cb_data.T_curr_ref->data(0, 3),
        -cb_data.T_curr_ref->data(1, 3), cb_data.T_curr_ref->data(0, 3), 0.0f;
    // F = K^{-T} * t_x * R * K^{-1}
    Eigen::Matrix3f K_inv = K.inverse();
    Eigen::Matrix3f F = K_inv.transpose()*t_x*R*K_inv;
    // Vector of parameters (a, b, c) of the epipolar line in curr image in implicit form ax+by+c=0
    Eigen::Vector3f l = F * Eigen::Vector3f((float)x, (float)y, 1.0f);
    const float rows = static_cast<float>(cb_data.curr_img->rows);
    const float cols = static_cast<float>(cb_data.curr_img->cols);
    // Clipping to image
    const float x_min = std::min(cols, std::max(0.0f, -l(2)/l(0))); // x coord of intersection with y=0
    const float x_max = std::min(cols, std::max(0.0f, (-l(2)-l(1)*rows)/l(0))); // x coord of intersection with y=rows
    cv::Point p1(x_min, (-l(2)-l(0)*x_min)/l(1)); // y= (-c-ax)/b
    cv::Point p2(x_max, (-l(2)-l(0)*x_max)/l(1));
    // Plot clipped epipolar line
    cv::line(*cb_data.curr_img, p1, p2, cv::Scalar(255), 1);
    cv::circle(*cb_data.curr_img, p1, 5, cv::Scalar(255));
    cv::circle(*cb_data.curr_img, p2, 5, cv::Scalar(255));

    cv::imshow("curr", *cb_data.curr_img);
  }
}

TEST(RMDCuTests, epipolarTest)
{
  rmd::PinholeCamera cam(481.2f, -480.0f, 319.5f, 239.5f);

  rmd::test::Dataset dataset("first_200_frames_traj_over_table_input_sequence.txt");
  if(!dataset.loadPathFromEnv())
  {
    FAIL() << "could not retrieve dataset path from the environment variable '"
           << rmd::test::Dataset::getDataPathEnvVar() <<"'" << std::endl;
  }
  if (!dataset.readDataSequence())
    FAIL() << "could not read dataset";

  const size_t ref_ind = 1;
  const size_t curr_ind = 199;

  const auto ref_entry = dataset(ref_ind);
  cv::Mat ref_img;
  dataset.readImage(ref_img, ref_entry);
  cv::Mat ref_depthmap;
  dataset.readDepthmap(ref_depthmap, ref_entry, ref_img.cols, ref_img.rows);
  rmd::SE3<float> T_world_ref;
  dataset.readCameraPose(T_world_ref, ref_entry);

  const auto curr_entry = dataset(curr_ind);
  cv::Mat curr_img;
  dataset.readImage(curr_img, curr_entry);
  rmd::SE3<float> T_world_curr;
  dataset.readCameraPose(T_world_curr, curr_entry);

  rmd::SE3<float> T_curr_ref = T_world_curr.inv() * T_world_ref;

  CallbackData cb_data;
  cb_data.ref_img  = &ref_img;
  cb_data.curr_img = &curr_img;
  cb_data.ref_depthmap = &ref_depthmap;
  cb_data.T_curr_ref = &T_curr_ref;
  cb_data.cam = &cam;

  cv::Mat colored_ref_repthmap = rmd::Depthmap::scaleMat(ref_depthmap);

  cv::imshow("ref",  ref_img);
  cv::imshow("curr", curr_img);
  cv::imshow("depthmap", colored_ref_repthmap);
  cv::setMouseCallback("ref", mouseCallback, &cb_data);
  cv::waitKey();
}

TEST(RMDCuTests, epipolarMatchTest)
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
  const size_t curr_ind = 1;

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
        reinterpret_cast<float*>(curr_img_flt.data),
        T_world_curr.inv());

  sdkStopTimer(&timer);
  double t = sdkGetAverageTimerValue(&timer) / 1000.0;
  printf("update CUDA execution time: %f seconds.\n", t);

  float2 * epipolar_matches = new float2[ref_img.cols * ref_img.rows];
  seeds.downloadEpipolarMatches(epipolar_matches);

  int * cu_convergence = new int[ref_img.cols * ref_img.rows];
  seeds.downloadConvergence(cu_convergence);

  for(size_t r=0; r<ref_img.rows; ++r)
  {
    for(size_t c=0; c<ref_img.cols; ++c)
    {
      const float match_x = epipolar_matches[ref_img.cols*r+c].x;
      const float match_y = epipolar_matches[ref_img.cols*r+c].y;
      const int convergence = cu_convergence[ref_img.cols*r+c];
      if(rmd::ConvergenceStates::UPDATE == convergence)
      {
        EXPECT_NEAR(static_cast<float>(c), match_x, 0.01);
        EXPECT_NEAR(static_cast<float>(r), match_y, 0.01);
      }
      // printf("Pixel coordinates: (reference image) (%lu, %lu) -> (%f, %f) (current image)\n", c, r, match_x, match_y);
    }
  }

  delete cu_convergence;
  delete epipolar_matches;

}
