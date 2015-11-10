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
      float k1,
      float k2,
      float r1,
      float r2);

  bool setReferenceImage(
      const cv::Mat &img_curr,
      const SE3<float> &T_curr_world,
      const float &min_depth,
      const float &max_depth);

  void update(const cv::Mat &img_curr,
              const SE3<float> &T_curr_world);

  void outputDepthmap(cv::Mat &depth_32fc1) const;
  void outputDenoisedDepthmap(cv::Mat &depth_32fc1, float lambda, int iterations);
  size_t getConvergedCount() const;
  float  getConvergedPercentage() const;

  static cv::Mat scaleMat(const cv::Mat &depthmap);

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
