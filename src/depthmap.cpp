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
  denoiser_.reset(new rmd::DepthmapDenoiser(width_, height_));
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
  denoiser_->setLargeSigmaSq(max_depth-min_depth);
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

void rmd::Depthmap::outputDepthmap(cv::Mat &depth_32fc1) const
{
  depth_32fc1.create(height_, width_, CV_32FC1);
  seeds_.downloadDepthmap(reinterpret_cast<float*>(depth_32fc1.data));
}

void rmd::Depthmap::outputDenoisedDepthmap(cv::Mat &depth_32fc1, float lambda, int iterations)
{
  depth_32fc1.create(height_, width_, CV_32FC1);
  denoiser_->denoise(
        seeds_.getMu(),
        seeds_.getSigmaSq(),
        seeds_.getA(),
        seeds_.getB(),
        reinterpret_cast<float*>(depth_32fc1.data),
        lambda,
        iterations);
}
