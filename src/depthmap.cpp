#include <rmd/depthmap.h>

rmd::Depthmap::Depthmap(
    size_t width,
    size_t height,
    float fx,
    float cx,
    float fy,
    float cy)
  : width_(width)
  , height_(height)
  , is_distorted_(false)
  , seeds_(width, height, rmd::PinholeCamera(fx, fy, cx, cy))
{
  cv_K_ = (cv::Mat_<float>(3, 3) << fx, 0.0f, cx, 0.0f, fy, cy, 0.0f, 0.0f, 1.0f);
}

void rmd::Depthmap::initUndistortionMap(
    float d0,
    float d1,
    float d2,
    float d3,
    float d4)
{
  cv_D_ = (cv::Mat_<float>(1, 5) << d0, d1, d2, d3, d4);
  cv::initUndistortRectifyMap(
        cv_K_,
        cv_D_,
        cv::Mat_<double>::eye(3,3),
        cv_K_,
        cv::Size(width_, height_),
        CV_16SC2,
        undist_map1_, undist_map2_);
  is_distorted_ = true;
}

bool rmd::Depthmap::setReferenceImage(
    const cv::Mat &img_curr,
    const SE3<float> &T_curr_world,
    const float &min_depth,
    const float &max_depth)
{
  inputImage(img_curr);
  return seeds_.setReferenceImage(
        reinterpret_cast<float*>(img_undistorted_32fc1_.data),
        T_curr_world,
        min_depth,
        max_depth);
}

void rmd::Depthmap::update(
    const cv::Mat &img_curr,
    const SE3<float> &T_curr_world)
{
  inputImage(img_curr);
  seeds_.update(
        reinterpret_cast<float*>(img_undistorted_32fc1_.data),
        T_curr_world);
}

void rmd::Depthmap::inputImage(const cv::Mat &img_8uc1)
{
  cv::Mat img_undistorted_8uc1;
  if(is_distorted_)
  {
    cv::remap(img_8uc1, img_undistorted_8uc1, undist_map1_, undist_map2_, CV_INTER_LINEAR);
  }
  else
  {
    img_undistorted_8uc1 = img_8uc1;
  }
  img_undistorted_8uc1.convertTo(img_undistorted_32fc1_, CV_32F, 1.0f/255.0f);
}

void rmd::Depthmap::outputDepthmap(cv::Mat &depth_32fc1)
{
  depth_32fc1.create(height_, width_, CV_32FC1);
  seeds_.downloadDepthmap(reinterpret_cast<float*>(depth_32fc1.data));
}

#ifdef RMD_DEBUG
void rmd::Depthmap::outputDisparity(cv::Mat &depth_32fc1_x, cv::Mat &depth_32fc1_y)
{
  depth_32fc1_x.create(height_, width_, CV_32FC1);
  depth_32fc1_y.create(height_, width_, CV_32FC1);
  seeds_.downloadDisparity(
        reinterpret_cast<float*>(depth_32fc1_x.data),
        reinterpret_cast<float*>(depth_32fc1_y.data));
}

void rmd::Depthmap::outputConvergence(cv::Mat &conv_8uc1)
{
  conv_8uc1.create(height_, width_, CV_8UC1);
  seeds_.downloadConvergence(reinterpret_cast<unsigned char*>(conv_8uc1.data));
}
#endif
