#ifndef DEPTHMAP_H
#define DEPTHMAP_H

#include <Eigen/Eigen>
#include <opencv2/imgproc/imgproc.hpp>

#include <rmd/seed_matrix.cuh>

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

  void setReferenceImage(
      const cv::Mat &img_curr,
      const SE3<float> &T_curr_world);

  void update(const cv::Mat &img_curr,
      const SE3<float> &T_curr_world);

private:
  void inputImage(const cv::Mat &img_8uc1);

  SeedMatrix seeds;

  size_t m_width;
  size_t m_height;
  float m_intrinsics[4];  // K matrix stored row-major

  cv::Mat m_cv_K, m_cv_D;
  cv::Mat m_undist_map1, m_undist_map2;
  cv::Mat m_img_undistorted_32fc1;
  bool m_is_distorted;
};

}

#endif // DEPTHMAP_H
