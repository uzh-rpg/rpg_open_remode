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
  colored_.create(depthmap->getHeight(), depthmap_->getWidth(), CV_8UC3);
  image_transport::ImageTransport it(nh_);
  depthmap_publisher_ = it.advertise("remode/depth",       10);
  conv_publisher_     = it.advertise("remode/convergence", 10);
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
    const rmd::SE3<float> T_world_ref = depthmap_->getT_world_ref();

    const float fx = depthmap_->getFx();
    const float fy = depthmap_->getFy();
    const float cx = depthmap_->getCx();
    const float cy = depthmap_->getCy();

    for(int y=0; y<depth.rows; ++y)
    {
      for(int x=0; x<depth.cols; ++x)
      {
        const float3 f = normalize( make_float3((x-cx)/fx, (y-cy)/fy, 1.0f) );
        const float3 xyz = T_world_ref * ( f * depth.at<float>(y, x) );
        if( rmd::ConvergenceState::CONVERGED == convergence.at<int>(y, x) )
        {
          PointType p;
          p.x = xyz.x;
          p.y = xyz.y;
          p.z = xyz.z;
          const uint8_t intensity = ref_img.at<uint8_t>(y, x);
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
  }
}

void rmd::Publisher::publishDepthmapAndPointCloud() const
{
  publishDepthmap();
  publishPointCloud();
}

void rmd::Publisher::publishConvergenceMap()
{
  std::lock_guard<std::mutex> lock(depthmap_->getRefImgMutex());

  const cv::Mat convergence = depthmap_->getConvergenceMap();
  const cv::Mat ref_img = depthmap_->getReferenceImage();

  cv::cvtColor(ref_img, colored_, CV_GRAY2BGR);
  for (int r = 0; r < colored_.rows; r++)
  {
    for (int c = 0; c < colored_.cols; c++)
    {
      switch(convergence.at<int>(r, c))
      {
      case rmd::ConvergenceState::CONVERGED:
        colored_.at<cv::Vec3b>(r, c)[0] = 255;
        break;
      case rmd::ConvergenceState::DIVERGED:
        colored_.at<cv::Vec3b>(r, c)[2] = 255;
        break;
      default:
        break;
      }
    }
  }
  cv_bridge::CvImage cv_image;
  cv_image.header.frame_id = "convergence_map";
  cv_image.encoding = sensor_msgs::image_encodings::BGR8;
  cv_image.image = colored_;
  if(nh_.ok())
  {
    cv_image.header.stamp = ros::Time::now();
    conv_publisher_.publish(cv_image.toImageMsg());
    std::cout << "INFO: publishing convergence map" << std::endl;
  }
}
