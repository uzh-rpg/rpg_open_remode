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

#include <rmd/depthmap_node.h>

#include <rmd/se3.cuh>

#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <vikit/params_helper.h>

#include <future>

rmd::DepthmapNode::DepthmapNode(ros::NodeHandle &nh)
  : nh_(nh)
  , num_msgs_(0)
{
  state_ = rmd::State::TAKE_REFERENCE_FRAME;
}

bool rmd::DepthmapNode::init()
{
  if(!vk::hasParam("remode/cam_width"))
    return false;
  if(!vk::hasParam("remode/cam_height"))
    return false;
  if(!vk::hasParam("remode/cam_fx"))
    return false;
  if(!vk::hasParam("remode/cam_fy"))
    return false;
  if(!vk::hasParam("remode/cam_cx"))
    return false;
  if(!vk::hasParam("remode/cam_cy"))
    return false;

  const size_t cam_width  = vk::getParam<int>("remode/cam_width");
  const size_t cam_height = vk::getParam<int>("remode/cam_height");
  const float  cam_fx     = vk::getParam<float>("remode/cam_fx");
  const float  cam_fy     = vk::getParam<float>("remode/cam_fy");
  const float  cam_cx     = vk::getParam<float>("remode/cam_cx");
  const float  cam_cy     = vk::getParam<float>("remode/cam_cy");

  depthmap_ = std::make_shared<rmd::Depthmap>(cam_width,
                                              cam_height,
                                              cam_fx,
                                              cam_cx,
                                              cam_fy,
                                              cam_cy);

  if(vk::hasParam("remode/cam_k1") &&
     vk::hasParam("remode/cam_k2") &&
     vk::hasParam("remode/cam_r1") &&
     vk::hasParam("remode/cam_r2") )
  {
    depthmap_->initUndistortionMap(
          vk::getParam<float>("remode/cam_k1"),
          vk::getParam<float>("remode/cam_k2"),
          vk::getParam<float>("remode/cam_r1"),
          vk::getParam<float>("remode/cam_r2"));
  }

  ref_compl_perc_    = vk::getParam<float>("remode/ref_compl_perc",   10.0f);
  max_dist_from_ref_ = vk::getParam<float>("remode/max_dist_from_ref", 0.5f);
  publish_conv_every_n_ = vk::getParam<float>("remode/publish_conv_every_n", 10);

  publisher_.reset(new rmd::Publisher(nh_, depthmap_));

  return true;
}

void rmd::DepthmapNode::denseInputCallback(
    const svo_msgs::DenseInputConstPtr &dense_input)
{
  num_msgs_ += 1;
  if(!depthmap_)
  {
    ROS_ERROR("depthmap not initialized. Call the DepthmapNode::init() method");
    return;
  }
  cv::Mat img_8uC1;
  try
  {
    cv_bridge::CvImageConstPtr cv_img_ptr =
        cv_bridge::toCvShare(dense_input->image,
                             dense_input,
                             sensor_msgs::image_encodings::MONO8);
    img_8uC1 = cv_img_ptr->image;
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
    if(depthmap_->setReferenceImage(
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
    depthmap_->update(img_8uC1, T_world_curr.inv());
    const float perc_conv = depthmap_->getConvergedPercentage();
    const float dist_from_ref = depthmap_->getDistFromRef();
    std::cout << "INFO: percentage of converged measurements: " << perc_conv << "%" << std::endl;
    if(perc_conv > ref_compl_perc_ || dist_from_ref > max_dist_from_ref_)
    {
      state_ = State::TAKE_REFERENCE_FRAME;
      denoiseAndPublishResults();
    }
    break;
  }
  default:
    break;
  }
  if(publish_conv_every_n_ < num_msgs_)
  {
    publishConvergenceMap();
    num_msgs_ = 0;
  }
}

void rmd::DepthmapNode::denoiseAndPublishResults()
{
  depthmap_->downloadDenoisedDepthmap(0.5f, 200);
  depthmap_->downloadConvergenceMap();

  std::async(std::launch::async,
             &rmd::Publisher::publishDepthmapAndPointCloud,
             *publisher_);
}

void rmd::DepthmapNode::publishConvergenceMap()
{
  depthmap_->downloadConvergenceMap();

  std::async(std::launch::async,
             &rmd::Publisher::publishConvergenceMap,
             *publisher_);
}
