#include <rmd/depthmap.h>

rmd::Depthmap::Depthmap(
    size_t width,
    size_t height,
    float fx,
    float cx,
    float fy,
    float cy)
  : m_width(width)
  , m_height(height)
  , m_is_distorted(false)
  , m_seeds(width, height, rmd::PinholeCamera(fx, fy, cx, cy))
{
  m_cv_K = (cv::Mat_<float>(3, 3) << fx, 0.0f, cx, 0.0f, fy, cy, 0.0f, 0.0f, 1.0f);
}

void rmd::Depthmap::initUndistortionMap(
    float d0,
    float d1,
    float d2,
    float d3,
    float d4)
{
  m_cv_D = (cv::Mat_<float>(1, 5) << d0, d1, d2, d3, d4);
  cv::initUndistortRectifyMap(m_cv_K, m_cv_D, cv::Mat_<double>::eye(3,3), m_cv_K,
                              cv::Size(m_width, m_height), CV_16SC2,
                              m_undist_map1, m_undist_map2);
  m_is_distorted = true;
}

bool rmd::Depthmap::setReferenceImage(
    const cv::Mat &img_curr,
    const SE3<float> &T_curr_world,
    const float &min_depth,
    const float &max_depth)
{
  inputImage(img_curr);
  return m_seeds.setReferenceImage(
        reinterpret_cast<float*>(m_img_undistorted_32fc1.data),
        T_curr_world,
        min_depth,
        max_depth);
}

void rmd::Depthmap::update(
    const cv::Mat &img_curr,
    const SE3<float> &T_curr_world)
{
  inputImage(img_curr);
  m_seeds.update(
        reinterpret_cast<float*>(m_img_undistorted_32fc1.data),
        T_curr_world);
}

void rmd::Depthmap::inputImage(const cv::Mat &img_8uc1)
{
  cv::Mat img_undistorted_8uc1;
  if(m_is_distorted)
  {
    cv::remap(img_8uc1, img_undistorted_8uc1, m_undist_map1, m_undist_map2, CV_INTER_LINEAR);
  }
  else
  {
    img_undistorted_8uc1 = img_8uc1;
  }
  img_undistorted_8uc1.convertTo(m_img_undistorted_32fc1, CV_32F, 1.0f/255.0f);
}

void rmd::Depthmap::outputDepthmap(cv::Mat &depth_32fc1)
{
  depth_32fc1.create(m_height, m_width, CV_32FC1);
  m_seeds.downloadDepthmap(reinterpret_cast<float*>(depth_32fc1.data));
}

#ifdef RMD_DEBUG
void rmd::Depthmap::outputDisparity(cv::Mat &depth_32fc1_x, cv::Mat &depth_32fc1_y)
{
  depth_32fc1_x.create(m_height, m_width, CV_32FC1);
  depth_32fc1_y.create(m_height, m_width, CV_32FC1);
  m_seeds.downloadDisparity(
        reinterpret_cast<float*>(depth_32fc1_x.data),
        reinterpret_cast<float*>(depth_32fc1_y.data));
}

void rmd::Depthmap::outputConvergence(cv::Mat &conv_8uc1)
{
  conv_8uc1.create(m_height, m_width, CV_8UC1);
  m_seeds.downloadConvergence(reinterpret_cast<unsigned char*>(conv_8uc1.data));
}

#endif
