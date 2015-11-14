#include <rmd/publisher.h>

#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

rmd::Publisher::Publisher(float fx, float cx, float fy, float cy, ros::NodeHandle &nh)
  : fx_(fx)
  , cx_(cx)
  , fy_(fy)
  , cy_(cy)
  , nh_(nh)
{
  image_transport::ImageTransport it(nh_);
  depthmap_publisher_= it.advertise("remode/depth", 10);
}

void rmd::Publisher::publishDepthmap(const cv::Mat &depthmap)
{
  cv_bridge::CvImage cv_image;
  cv_image.header.stamp = ros::Time::now();
  cv_image.header.frame_id = "depthmap";
  cv_image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
  cv_image.image = depthmap;
  depthmap_publisher_.publish(cv_image.toImageMsg());
}
