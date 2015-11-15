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

private:
  ros::NodeHandle &nh_;
  std::shared_ptr<rmd::Depthmap> depthmap_;

  image_transport::Publisher depthmap_publisher_;

  PointCloud::Ptr pc_;
  ros::Publisher pub_pc_;
};

} // rmd namespace

#endif // PUBLISHER_H
