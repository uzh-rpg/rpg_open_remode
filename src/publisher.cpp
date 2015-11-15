#include <rmd/publisher.h>

#include <rmd/seed_matrix.cuh>

#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

#include <Eigen/Eigen>

rmd::Publisher::Publisher(ros::NodeHandle &nh,
                          std::shared_ptr<rmd::Depthmap> depthmap)
  : nh_(nh)
  , pc_(new PointCloud)
{
  depthmap_ = depthmap;
  image_transport::ImageTransport it(nh_);
  depthmap_publisher_= it.advertise("remode/depth", 10);
  pub_pc_ = nh_.advertise<PointCloud>("remode/pointcloud", 1);
}

void rmd::Publisher::publishDepthmap() const
{
  cv_bridge::CvImage cv_image;
  cv_image.header.frame_id = "depthmap";
  cv_image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
  cv_image.image = depthmap_->getDepthmap();
  if(nh_.ok())
  {
    cv_image.header.stamp = ros::Time::now();
    depthmap_publisher_.publish(cv_image.toImageMsg());
    std::cout << "INFO: publishing depth map" << std::endl;
  }
}

void rmd::Publisher::publishPointCloud() const
{
  {
    std::lock_guard<std::mutex> lock(depthmap_->getRefImgMutex());

    const cv::Mat depth = depthmap_->getDepthmap();
    const cv::Mat convergence = depthmap_->getConvergenceMap();
    const cv::Mat ref_img = depthmap_->getReferenceImage();

    const float fx = depthmap_->getFx();
    const float fy = depthmap_->getFy();
    const float cx = depthmap_->getCx();
    const float cy = depthmap_->getCy();

    for(int y=0; y<depth.rows; ++y)
    {
      for(int x=0; x<depth.cols; ++x)
      {
        Eigen::Vector3f f((x-cx)/fx, (y-cy)/fy, 1.0);
        f.normalize();
        Eigen::Vector3f xyz = f*depth.at<float>(y,x);
        if(convergence.at<int>(y, x) != rmd::ConvergenceState::DIVERGED &&
           convergence.at<int>(y, x) != rmd::ConvergenceState::BORDER)
        {
          PointType p;
          p.x = xyz.x();
          p.y = xyz.y();
          p.z = xyz.z();
          uint8_t intensity = ref_img.at<uint8_t>(y,x);
          p.intensity = intensity;
          pc_->push_back(p);
        }
      }
    }
  }
  if (!pc_->empty())
  {
    if(nh_.ok())
    {
      uint64_t timestamp;
#if PCL_MAJOR_VERSION == 1 && PCL_MINOR_VERSION >= 7
      pcl_conversions::toPCL(ros::Time::now(), timestamp);
#else
      timestamp = ros::Time::now();
#endif
      pc_->header.frame_id = "/world";

      pc_->header.stamp = timestamp;
      pub_pc_.publish(pc_);
      std::cout << "INFO: publishing pointcloud, " << pc_->size() << " points" << std::endl;
    }
    pc_->clear();
  }
}

void rmd::Publisher::publishDepthmapAndPointCloud() const
{
  publishDepthmap();
  publishPointCloud();
}
