#include <string>
#include <iostream>
#include <fstream>

#include <boost/filesystem.hpp>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

#include <rmd/depthmap.h>

struct DatasetEntry
{
  std::string file_name;
  Eigen::Vector3f translation;
  Eigen::Quaternionf quaternion;
};

bool readDataSequence(
    const std::string &data_file_path,
    std::vector<DatasetEntry> &dataset
    )
{
  dataset.clear();
  std::string line;
  std::ifstream sequence_file_str(data_file_path);
  if (sequence_file_str.is_open())
  {
    while (getline(sequence_file_str, line))
    {
      std::stringstream line_str(line);
      DatasetEntry data;
      line_str >> data.file_name;
      line_str >> data.translation.x();
      line_str >> data.translation.y();
      line_str >> data.translation.z();
      line_str >> data.quaternion.x();
      line_str >> data.quaternion.y();
      line_str >> data.quaternion.z();
      line_str >> data.quaternion.w();
      dataset.push_back(data);
    }
    sequence_file_str.close();
    return true;
  }
  else
  {
    return false;
  }
}

int main(int argc, char **argv)
{
  boost::filesystem::path dataset_path("/media/mpi/Elements/thinkpad_backup/matia/workspace/rpg_reconstruction/benchmark/datasets/traj_over_table");
  boost::filesystem::path sequence_file_path("/media/mpi/Elements/thinkpad_backup/matia/workspace/rpg_reconstruction/benchmark/monocular_depth/experiments/first_200_frames_traj_over_table_our_input_sequence.txt");

  std::vector<DatasetEntry> dataset;
  if (!readDataSequence(sequence_file_path.string(), dataset))
  {
    std::cerr << "ERROR: could not read dataset" << std::endl;
    return -1;
  }

  bool first_img = true;
  rmd::Depthmap depthmap(640, 480, 481.2f, 319.5f, -480.0f, 239.5f);

  for (auto it = dataset.begin(); it != dataset.end(); ++it)
  {
    const auto data = *it;
    const auto img_file_path = dataset_path / data.file_name;
    cv::Mat img = cv::imread(img_file_path.string(), CV_LOAD_IMAGE_GRAYSCALE);
    if(img.data == NULL)
    {
      std::cerr << "ERROR: could not read image " << img_file_path.string() << std::endl;
      continue;
    }
    rmd::SE3<float> T_world_curr(
          data.quaternion.w(),
          data.quaternion.x(),
          data.quaternion.y(),
          data.quaternion.z(),
          data.translation.x(),
          data.translation.y(),
          data.translation.z()
          );

    std::cout << "RUN EXPERIMENT: inputting image " << img_file_path.string() <<  std::endl;
    std::cout << "T_world_curr:" << std::endl;
    std::cout << T_world_curr << std::endl;

    // process
    if(first_img)
    {
      if(depthmap.setReferenceImage(img, T_world_curr.inv(), 0.4f, 1.8f))
      {
        first_img = false;
      }
      else
      {
        std::cerr << "ERROR: could not set reference image" << std::endl;
        return EXIT_FAILURE;
      }
    }
    else
    {
      depthmap.update(img, T_world_curr.inv());
    }
  }

  // denoise

  // show depthmap
  cv::Mat result;
  depthmap.outputDepthmap(result);
  cv::imshow("result", result);
  cv::waitKey();

  cv::imwrite("/home/mpi/Desktop/result.png", result);

  return EXIT_SUCCESS;
}
