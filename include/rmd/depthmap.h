#ifndef RMD_DEPTHMAP_H
#define RMD_DEPTHMAP_H

#include <memory>
#include <Eigen/Eigen>
#include <opencv2/imgproc/imgproc.hpp>

#include <rmd/seed_matrix.cuh>
#include <rmd/depthmap_denoiser.cuh>

namespace rmd
{

class Depthmap
{
public:
  Depthmap(
      size_t width,
      size_t height,
      float fx,
      float cx,
      float fy,
      float cy);

  void initUndistortionMap(
      float d0,
      float d1,
      float d2,
      float d3,
      float d4);

  bool setReferenceImage(
      const cv::Mat &img_curr,
      const SE3<float> &T_curr_world,
      const float &min_depth,
      const float &max_depth);

  void update(const cv::Mat &img_curr,
              const SE3<float> &T_curr_world);

  void outputDepthmap(cv::Mat &depth_32fc1) const;
  void outputDenoisedDepthmap(cv::Mat &depth_32fc1, float lambda, int iterations);

private:
  void inputImage(const cv::Mat &img_8uc1);

  SeedMatrix seeds_;

  size_t width_;
  size_t height_;

  cv::Mat cv_K_, cv_D_;
  cv::Mat undist_map1_, undist_map2_;
  cv::Mat img_undistorted_32fc1_;
  bool is_distorted_;

  std::unique_ptr<DepthmapDenoiser> denoiser_;
};

}

#endif // RMD_DEPTHMAP_H
