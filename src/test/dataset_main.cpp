#include <string>
#include <iostream>
#include <fstream>

#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

#include <rmd/depthmap.h>
#include "../test/dataset.h"

int main(int argc, char **argv)
{
  const boost::filesystem::path dataset_path("../test_data/images");
  const boost::filesystem::path sequence_file_path("../test_data/first_200_frames_traj_over_table_input_sequence.txt");

  rmd::test::Dataset dataset(dataset_path.string(), sequence_file_path.string());
  if (!dataset.readDataSequence())
  {
    std::cerr << "ERROR: could not read dataset" << std::endl;
    return -1;
  }

  bool first_img = true;
  rmd::Depthmap depthmap(640, 480, 481.2f, 319.5f, -480.0f, 239.5f);

  for(const auto data : dataset)
  {
    cv::Mat img;
    if(!dataset.readImage(img, data))
    {
      std::cerr << "ERROR: could not read image " << data.getFileName() << std::endl;
      continue;
    }
    rmd::SE3<float> T_world_curr;
    dataset.readCameraPose(T_world_curr, data);

    std::cout << "RUN EXPERIMENT: inputting image " << data.getFileName() <<  std::endl;
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
