
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
#include <svo_msgs/DenseInput.h>
#include "dataset.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "dataset_publisher");

  rmd::test::Dataset dataset("first_200_frames_traj_over_table_input_sequence.txt");
  if(!dataset.loadPathFromEnv())
  {
    std::cerr << "ERROR: could not retrieve dataset path from the environment variable '"
              << rmd::test::Dataset::getDataPathEnvVar() <<"'" << std::endl;
  }
  if (!dataset.readDataSequence(0, 200))
  {
    std::cerr << "ERROR: could not read dataset" << std::endl;
    return EXIT_FAILURE;
  }

  ros::NodeHandle nh;
  ros::Publisher publisher = nh.advertise<svo_msgs::DenseInput>(
        "/svo/dense_input",
        1);

  ros::Rate loop_rate(10);
  for(const auto data : dataset)
  {
    if(!ros::ok())
    {
      break;
    }
    cv::Mat img_8uC1;
    if(!dataset.readImage(img_8uC1, data))
    {
      std::cerr << "ERROR: could not read image " << data.getImageFileName() << std::endl;
      continue;
    }

    cv::Mat depth_32FC1;
    if(!dataset.readDepthmap(depth_32FC1, data, img_8uC1.cols, img_8uC1.rows))
    {
      std::cerr << "ERROR: could not read depthmap " << data.getDepthmapFileName() << std::endl;
      continue;
    }
    double min_depth, max_depth;
    cv::minMaxLoc(depth_32FC1, &min_depth, &max_depth);

    rmd::SE3<float> T_world_curr;
    dataset.readCameraPose(T_world_curr, data);

    std::cout << "DATASET PUBLISHER: image " << data.getImageFileName() <<  std::endl;
    std::cout << "T_world_curr:" << std::endl;
    std::cout << T_world_curr << std::endl;

    svo_msgs::DenseInput msg;
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = "/dense_input_frame_id";

    cv_bridge::CvImage cv_image;
    cv_image.header.stamp = ros::Time::now();
    cv_image.header.frame_id = "/greyscale_image_frame_id";
    cv_image.encoding = sensor_msgs::image_encodings::MONO8;
    cv_image.image = img_8uC1;
    msg.image = *cv_image.toImageMsg();

    msg.pose.orientation.w = data.getQuaternion().w();
    msg.pose.orientation.x = data.getQuaternion().x();
    msg.pose.orientation.y = data.getQuaternion().y();
    msg.pose.orientation.z = data.getQuaternion().z();

    msg.pose.position.x = data.getTranslation().x();
    msg.pose.position.y = data.getTranslation().y();
    msg.pose.position.z = data.getTranslation().z();

    msg.min_depth = min_depth;
    msg.max_depth = max_depth;

    publisher.publish(msg);

    loop_rate.sleep();
  }

  return EXIT_SUCCESS;
}
