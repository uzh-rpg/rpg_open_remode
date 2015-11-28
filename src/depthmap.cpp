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

#include <rmd/depthmap.h>

rmd::Depthmap::Depthmap(size_t width,
                        size_t height,
                        float fx,
                        float cx,
                        float fy,
                        float cy)
  : width_(width)
  , height_(height)
  , is_distorted_(false)
  , seeds_(width, height, rmd::PinholeCamera(fx, fy, cx, cy))
  , fx_(fx)
  , fy_(fy)
  , cx_(cx)
  , cy_(cy)
{
  cv_K_ = (cv::Mat_<float>(3, 3) << fx, 0.0f, cx, 0.0f, fy, cy, 0.0f, 0.0f, 1.0f);
  denoiser_.reset(new rmd::DepthmapDenoiser(width_, height_));

  output_depth_32fc1_ = cv::Mat_<float>(height_, width_);
  output_convergence_int_ = cv::Mat_<int>(height_, width_);
  img_undistorted_32fc1_.create(height_, width_, CV_32FC1);
  img_undistorted_8uc1_.create(height_, width_, CV_8UC1);
  ref_img_undistorted_8uc1_.create(height_, width_, CV_8UC1);
}

void rmd::Depthmap::initUndistortionMap(
    float k1,
    float k2,
    float r1,
    float r2)
{
  cv_D_ = (cv::Mat_<float>(1, 4) << k1, k2, r1, r2);
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
    const rmd::SE3<float> &T_curr_world,
    const float &min_depth,
    const float &max_depth)
{
  denoiser_->setLargeSigmaSq(max_depth-min_depth);
  inputImage(img_curr);
  const bool ret = seeds_.setReferenceImage(reinterpret_cast<float*>(img_undistorted_32fc1_.data),
                                            T_curr_world,
                                            min_depth,
                                            max_depth);

  {
    std::lock_guard<std::mutex> lock(ref_img_mutex_);
    img_undistorted_8uc1_.copyTo(ref_img_undistorted_8uc1_);
    T_world_ref_ = T_curr_world.inv();
  }

  return ret;
}

void rmd::Depthmap::update(
    const cv::Mat &img_curr,
    const rmd::SE3<float> &T_curr_world)
{
  inputImage(img_curr);
  seeds_.update(
        reinterpret_cast<float*>(img_undistorted_32fc1_.data),
        T_curr_world);
}

void rmd::Depthmap::inputImage(const cv::Mat &img_8uc1)
{
  if(is_distorted_)
  {
    cv::remap(img_8uc1, img_undistorted_8uc1_, undist_map1_, undist_map2_, CV_INTER_LINEAR);
  }
  else
  {
    img_undistorted_8uc1_ = img_8uc1;
  }
  img_undistorted_8uc1_.convertTo(img_undistorted_32fc1_, CV_32F, 1.0f/255.0f);
}

void rmd::Depthmap::downloadDepthmap()
{
  seeds_.downloadDepthmap(reinterpret_cast<float*>(output_depth_32fc1_.data));
}

void rmd::Depthmap::downloadDenoisedDepthmap(float lambda, int iterations)
{
  denoiser_->denoise(
        seeds_.getMu(),
        seeds_.getSigmaSq(),
        seeds_.getA(),
        seeds_.getB(),
        reinterpret_cast<float*>(output_depth_32fc1_.data),
        lambda,
        iterations);
}

const cv::Mat_<float> rmd::Depthmap::getDepthmap() const
{
  return output_depth_32fc1_;
}

void rmd::Depthmap::downloadConvergenceMap()
{
  seeds_.downloadConvergence(reinterpret_cast<int*>(output_convergence_int_.data));
}

const cv::Mat_<int> rmd::Depthmap::getConvergenceMap() const
{
  return output_convergence_int_;
}

const cv::Mat rmd::Depthmap::getReferenceImage() const
{
  return ref_img_undistorted_8uc1_;
}

size_t rmd::Depthmap::getConvergedCount() const
{
  return seeds_.getConvergedCount();
}

float rmd::Depthmap::getConvergedPercentage() const
{
  const size_t count = rmd::Depthmap::getConvergedCount();
  return static_cast<float>(count) / static_cast<float>(width_*height_) * 100.0f;
}

// Scale depth in [0,1] and cvt to color
// only for test and debug
cv::Mat rmd::Depthmap::scaleMat(const cv::Mat &depthmap)
{
  cv::Mat scaled_depthmap = depthmap.clone();
  double min_val, max_val;
  cv::minMaxLoc(scaled_depthmap, &min_val, &max_val);
  cv::Mat converted;
  scaled_depthmap = (scaled_depthmap - min_val) * 1.0 / (max_val - min_val);
  scaled_depthmap.convertTo(converted, CV_8UC1, 255);
  cv::Mat colored(converted.rows, converted.cols, CV_8UC3);
  cv::cvtColor(converted, colored, CV_GRAY2BGR);
  return colored;
}

float rmd::Depthmap::getDistFromRef() const
{
  return seeds_.getDistFromRef();
}
