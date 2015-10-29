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

#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <rmd/depthmap_node.h>
#include <rmd/se3.cuh>

rmd::DepthmapNode::DepthmapNode()
  : depthmap_(640, 480, 481.2f, 319.5f, -480.0f, 239.5f)
{
  state_ = rmd::State::TAKE_REFERENCE_FRAME;
}

void rmd::DepthmapNode::denseInputCallback(
    const svo_msgs::DenseInputConstPtr &dense_input)
{
  cv::Mat img_8uC1;
  try
  {
    img_8uC1 = cv_bridge::toCvCopy(
          dense_input->image,
          sensor_msgs::image_encodings::MONO8)->image;
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
  }
  rmd::SE3<float> T_world_curr(
        dense_input->pose.orientation.w,
        dense_input->pose.orientation.x,
        dense_input->pose.orientation.y,
        dense_input->pose.orientation.z,
        dense_input->pose.position.x,
        dense_input->pose.position.y,
        dense_input->pose.position.z);

  std::cout << "DEPTHMAP NODE: received image "
            << img_8uC1.cols << "x" << img_8uC1.rows
            <<  std::endl;
  std::cout << "T_world_curr:" << std::endl;
  std::cout << T_world_curr << std::endl;

  switch (state_) {
  case rmd::State::TAKE_REFERENCE_FRAME:
  {
    if(depthmap_.setReferenceImage(
         img_8uC1,
         T_world_curr.inv(),
         dense_input->min_depth,
         dense_input->max_depth))
    {
      state_ = State::UPDATE;
    }
    else
    {
      std::cerr << "ERROR: could not set reference image" << std::endl;
    }
    break;
  }
  case rmd::State::UPDATE:
  {
    depthmap_.update(img_8uC1, T_world_curr.inv());
#if 1
    cv::Mat curr_depth;
    depthmap_.outputDepthmap(curr_depth);
    cv::imshow("curr_depth", rmd::Depthmap::scaleMat(curr_depth));
    cv::waitKey(2);
#endif
    break;
  }
  default:
    break;
  }

}

