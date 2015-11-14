#ifndef RMD_PUBLISHER_H
#define RMD_PUBLISHER_H

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

  Publisher(float fx,
            float cx,
            float fy,
            float cy,
            ros::NodeHandle &nh);

  void publishDepthmap(const cv::Mat &depthmap) const;

  void publishPointCloud(const cv::Mat &depthmap,
                         const cv::Mat &ref_img,
                         const cv::Mat &convergence) const;

  void publishDepthmapAndPointCloud(const cv::Mat &depthmap,
                                    const cv::Mat &ref_img,
                                    const cv::Mat &convergence) const;

private:
  float fx_, cx_, fy_, cy_;

  ros::NodeHandle &nh_;
  image_transport::Publisher depthmap_publisher_;

  PointCloud::Ptr pc_;
  ros::Publisher pub_pc_;
};

} // rmd namespace

#endif // PUBLISHER_H
