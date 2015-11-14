#ifndef RMD_PUBLISHER_H
#define RMD_PUBLISHER_H

#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>

namespace rmd
{

class Publisher
{
public:

  Publisher(float fx,
            float cx,
            float fy,
            float cy,
            ros::NodeHandle &nh);

  void publishDepthmap(const cv::Mat &depthmap);

  void publishPointCloud(const cv::Mat &ref_img,
                         const cv::Mat &uncertainty);

private:
  float fx_, cx_, fy_, cy_;

  ros::NodeHandle &nh_;
  image_transport::Publisher depthmap_publisher_;
};

} // rmd namespace

#endif // PUBLISHER_H
