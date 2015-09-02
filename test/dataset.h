#ifndef RMD_TEST_DATASET_H
#define RMD_TEST_DATASET_H

#include <fstream>
#include <Eigen/Eigen>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <rmd/se3.cuh>

namespace rmd
{
namespace test
{

class DatasetEntry
{
public:
  const std::string & getImageFileName() const;
  std::string & getImageFileName();
  const std::string & getDepthmapFileName() const;
  std::string & getDepthmapFileName();
  const Eigen::Vector3f & getTranslation() const;
  Eigen::Vector3f & getTranslation();
  const Eigen::Quaternionf & getQuaternion() const;
  Eigen::Quaternionf & getQuaternion();

private:
  std::string image_file_name_;
  std::string depthmap_file_name_;
  Eigen::Vector3f translation_;
  Eigen::Quaternionf quaternion_;
};

class Dataset
{
public:
  Dataset(const std::string &dataset_path, const std::string &sequence_file_path);
  bool readDataSequence();
  bool readImage(cv::Mat &img, const DatasetEntry &entry) const;
  void readCameraPose(rmd::SE3<float> &pose, const DatasetEntry &entry) const;
  bool readDepthmap(
      cv::Mat &depthmap,
      const DatasetEntry &entry,
      const size_t &width,
      const size_t &height) const;
  std::vector<DatasetEntry>::const_iterator begin() const;
  std::vector<DatasetEntry>::const_iterator end() const;

private:
  std::string dataset_path_;
  std::string sequence_file_path_;
  std::vector<DatasetEntry> dataset_;
};

} // test namespace
} // rmd namespace

#endif // RMD_TEST_DATASET_H
