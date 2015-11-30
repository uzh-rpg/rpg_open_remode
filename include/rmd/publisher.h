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

#ifndef RMD_PUBLISHER_H
#define RMD_PUBLISHER_H

#include <rmd/depthmap.h>

#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>

#include <pcl_ros/point_cloud.h>

namespace rmd
{

class Publisher
{

  typedef pcl::PointXYZI PointType;
  typedef pcl::PointCloud<PointType> PointCloud;

public:

  Publisher(ros::NodeHandle &nh,
            std::shared_ptr<rmd::Depthmap> depthmap);

  void publishDepthmap() const;

  void publishPointCloud() const;

  void publishDepthmapAndPointCloud() const;

  void publishConvergenceMap();

private:

  ros::NodeHandle &nh_;
  std::shared_ptr<rmd::Depthmap> depthmap_;

  image_transport::Publisher depthmap_publisher_;
  image_transport::Publisher conv_publisher_;

  PointCloud::Ptr pc_;
  ros::Publisher pub_pc_;

  cv::Mat colored_;
};

} // rmd namespace

#endif // PUBLISHER_H
