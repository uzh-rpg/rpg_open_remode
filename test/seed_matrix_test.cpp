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

  StopWatchInterface * timer = NULL;
  sdkCreateTimer(&timer);
  sdkResetTimer(&timer);
  sdkStartTimer(&timer);

  seeds.setReferenceImage(reinterpret_cast<float*>(ref_img_flt.data), T_curr_world, min_scene_depth, max_scene_depth);

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

  int side = seeds.getPatchSide();
  for(size_t y=side; y<ref_img.rows-side/2; ++y)
  {
    for(size_t x=side; x<ref_img.cols-side/2; ++x)
    {
      double sum_templ    = 0.0f;
      double sum_templ_sq = 0.0f;
      for(int patch_y=0; patch_y<side; ++patch_y)
      {
        for(int patch_x=0; patch_x<side; ++patch_x)
        {
          const double templ = (double) ref_img_flt.at<float>( y-side/2+patch_y, x-side/2+patch_x );
          sum_templ += templ;
          sum_templ_sq += templ*templ;
        }
      }
      ocv_sum_templ.at<float>(y, x) = (float) sum_templ;
      ocv_const_templ_denom.at<float>(y, x) = (float) ( ((double)(side*side))*sum_templ_sq - sum_templ*sum_templ );
    }
  }
  for(size_t r=side; r<ref_img.rows-side/2; ++r)
  {
    for(size_t c=side; c<ref_img.cols-side/2; ++c)
    {
      ASSERT_NEAR(ocv_sum_templ.at<float>(r, c), cu_sum_templ.at<float>(r, c), 0.00001f);
      ASSERT_NEAR(ocv_const_templ_denom.at<float>(r, c), cu_const_templ_denom.at<float>(r, c), 0.001f);
    }
  }
}
