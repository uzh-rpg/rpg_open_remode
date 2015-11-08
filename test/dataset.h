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

#ifndef RMD_TEST_DATASET_H
#define RMD_TEST_DATASET_H

#include <fstream>
#include <Eigen/Eigen>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <rmd/se3.cuh>
#include <rmd/pinhole_camera.cuh>

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
  Dataset(
      const std::string &dataset_path,
      const std::string &sequence_file);
  Dataset(
      const std::string &sequence_file);
  Dataset();
  bool readDataSequence(size_t start, size_t end=0);
  bool readDataSequence();
  bool readImage(cv::Mat &img, const char *file_name) const;
  bool readImage(cv::Mat &img, const DatasetEntry &entry) const;
  void readCameraPose(rmd::SE3<float> &pose, const DatasetEntry &entry) const;
  bool readDepthmap(
      cv::Mat &depthmap,
      const DatasetEntry &entry,
      const size_t &width,
      const size_t &height) const;
  std::vector<DatasetEntry>::const_iterator begin() const;
  std::vector<DatasetEntry>::const_iterator end() const;
  const DatasetEntry & operator()(size_t index) const;

  bool loadPathFromEnv();
  static const char * getDataPathEnvVar();

private:
  std::string dataset_path_;
  const std::string sequence_file_;
  std::vector<DatasetEntry> dataset_;
  const rmd::PinholeCamera cam_;

  static constexpr const char *data_path_env_var = "RMD_TEST_DATA_PATH";
};

} // test namespace
} // rmd namespace

#endif // RMD_TEST_DATASET_H
