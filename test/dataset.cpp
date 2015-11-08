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

#include "dataset.h"

const std::string & rmd::test::DatasetEntry::getImageFileName() const
{
  return image_file_name_;
}

std::string & rmd::test::DatasetEntry::getImageFileName()
{
  return image_file_name_;
}

const std::string & rmd::test::DatasetEntry::getDepthmapFileName() const
{
  return depthmap_file_name_;
}

std::string & rmd::test::DatasetEntry::getDepthmapFileName()
{
  return depthmap_file_name_;
}

const Eigen::Vector3f & rmd::test::DatasetEntry::getTranslation() const
{
  return translation_;
}

Eigen::Vector3f & rmd::test::DatasetEntry::getTranslation()
{
  return translation_;
}

const Eigen::Quaternionf & rmd::test::DatasetEntry::getQuaternion() const
{
  return quaternion_;
}

Eigen::Quaternionf & rmd::test::DatasetEntry::getQuaternion()
{
  return quaternion_;
}

rmd::test::Dataset::Dataset(
    const std::string &dataset_path,
    const std::string &sequence_file)
  : dataset_path_(dataset_path)
  , sequence_file_(sequence_file)
{
}

rmd::test::Dataset::Dataset(
    const std::string &sequence_file)
  : dataset_path_(std::string())
  , sequence_file_(sequence_file)
{
}

rmd::test::Dataset::Dataset()
  : dataset_path_(std::string())
  , sequence_file_(std::string())
{
}

bool rmd::test::Dataset::readDataSequence(size_t start, size_t end)
{
  if(dataset_path_.empty() || sequence_file_.empty())
  {
    return false;
  }
  dataset_.clear();
  std::string line;
  const auto sequence_file_path = boost::filesystem::path(dataset_path_) / sequence_file_;
  std::ifstream sequence_file_str(sequence_file_path.string());
  if (sequence_file_str.is_open())
  {
    size_t line_cnt = 0;
    while (getline(sequence_file_str, line))
    {
      if(line_cnt >= start)
      {
        if(line_cnt < end || 0 == end)
        {
          std::stringstream line_str(line);
          DatasetEntry data;
          std::string imgFileName;
          line_str >> imgFileName;
          data.getImageFileName() = imgFileName;
          const std::string depthmapFileName = imgFileName.substr(0, imgFileName.find('.')+1) + "depth";
          data.getDepthmapFileName() = depthmapFileName;
          line_str >> data.getTranslation().x();
          line_str >> data.getTranslation().y();
          line_str >> data.getTranslation().z();
          line_str >> data.getQuaternion().x();
          line_str >> data.getQuaternion().y();
          line_str >> data.getQuaternion().z();
          line_str >> data.getQuaternion().w();
          dataset_.push_back(data);
        }
      }
      line_cnt += 1;
    }
    sequence_file_str.close();
    return true;
  }
  else
    return false;
}

bool rmd::test::Dataset::readDataSequence()
{
  return readDataSequence(0, 0);
}

bool rmd::test::Dataset::readImage(cv::Mat &img, const char *file_name) const
{
  const boost::filesystem::path dataset_path(dataset_path_);
  const auto img_file_path = dataset_path / "images" / file_name;
  img = cv::imread(img_file_path.string(), CV_LOAD_IMAGE_GRAYSCALE);
  if(img.data == NULL)
    return false;
  else
    return true;
}

bool rmd::test::Dataset::readImage(cv::Mat &img, const DatasetEntry &entry) const
{
  return readImage(img, entry.getImageFileName().c_str());
}

void rmd::test::Dataset::readCameraPose(rmd::SE3<float> &pose, const DatasetEntry &entry) const
{
  pose = rmd::SE3<float>(
        entry.getQuaternion().w(),
        entry.getQuaternion().x(),
        entry.getQuaternion().y(),
        entry.getQuaternion().z(),
        entry.getTranslation().x(),
        entry.getTranslation().y(),
        entry.getTranslation().z()
        );
}

bool rmd::test::Dataset::readDepthmap(
    cv::Mat &depthmap,
    const DatasetEntry &entry,
    const size_t &width,
    const size_t &height) const
{
  const boost::filesystem::path dataset_path(dataset_path_);
  const auto depthmap_file_path = dataset_path / "depthmaps" / entry.getDepthmapFileName();
  std::ifstream depthmap_file_str(depthmap_file_path.string());
  if (depthmap_file_str.is_open())
  {
    depthmap.create(height, width, CV_32FC1);
    float z;
    for(size_t r=0; r<height; ++r)
    {
      for(size_t c=0; c<width; ++c)
      {
        depthmap_file_str >> z;
        depthmap.at<float>(r, c) = z / 100.0f;
      }
    }
    depthmap_file_str.close();
    return true;
  }
  else
    return false;
}

std::vector<rmd::test::DatasetEntry>::const_iterator rmd::test::Dataset::begin() const
{ return dataset_.begin(); }

std::vector<rmd::test::DatasetEntry>::const_iterator rmd::test::Dataset::end() const
{ return dataset_.end(); }

const rmd::test::DatasetEntry & rmd::test::Dataset::operator()(size_t index) const
{
  return dataset_.at(index);
}

bool rmd::test::Dataset::loadPathFromEnv()
{
  const char *env_path = std::getenv(data_path_env_var);
  if(nullptr != env_path)
  {
    dataset_path_ = std::string(env_path);
    return true;
  }
  return false;
}

const char * rmd::test::Dataset::getDataPathEnvVar()
{
  return data_path_env_var;
}
