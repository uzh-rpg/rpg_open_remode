#include <gtest/gtest.h>
#include <cuda_toolkit/helper_timer.h>

#include <rmd/seed_matrix.cuh>
#include <rmd/se3.cuh>

#include <opencv2/opencv.hpp>

TEST(RMDCuTests, seedMatrixInit)
{
  rmd::PinholeCamera cam(481.2f, -480.0f, 319.5f, 239.5f);
  cv::Mat ref_img = cv::imread(
        "/media/mpi/Elements/thinkpad_backup/matia/workspace/rpg_reconstruction/benchmark/datasets/traj_over_table/scene_000.png",
        CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat ref_img_flt;
  ref_img.convertTo(ref_img_flt, CV_32F, 1.0f/255.0f);

  rmd::SE3<float> T_curr_world(
        0.611661f,
        0.789455f,
        0.051299f,
        -0.000779f,
        1.086410f,
        4.766730f,
        -1.449960f);
  const float min_scene_depth = 0.4f;
  const float max_scene_depth = 1.8f;

  rmd::SeedMatrix seeds(ref_img.cols, ref_img.rows, cam);
  seeds.setReferenceImage(reinterpret_cast<float*>(ref_img_flt.data), T_curr_world, min_scene_depth, max_scene_depth);

  cv::Mat initial_depthmap(ref_img.rows, ref_img.cols, CV_32FC1);
  seeds.downloadDepthmap(reinterpret_cast<float*>(initial_depthmap.data));

  for(size_t r=0; r<ref_img.rows; ++r)
  {
    for(size_t c=0; c<ref_img.cols; ++c)
    {
      ASSERT_FLOAT_EQ((min_scene_depth+max_scene_depth)/2.0f, initial_depthmap.at<float>(r, c));
    }
  }
}
