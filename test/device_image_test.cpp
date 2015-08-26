#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "copy.cuh"
#include "sobel.cuh"

TEST(RMDCuTests, deviceImageUploadDownloadFloat)
{
  cv::Mat img = cv::imread("/home/mpi/Desktop/pict/DSC_0182.JPG", CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat img_flt;
  img.convertTo(img_flt, CV_32F, 1./255.);

  const size_t w = img_flt.cols;
  const size_t h = img_flt.rows;
  // upload data to gpu memory
  rmd::DeviceImage<float> in_img(w, h);
  in_img.setDevData(reinterpret_cast<float*>(img_flt.data));

  float * cu_img = new float[w*h];
  in_img.getDevData(cu_img);

  for(size_t y=0; y<h; ++y)
  {
    for(size_t x=0; x<w; ++x)
    {
      ASSERT_FLOAT_EQ(img_flt.at<float>(y, x), cu_img[y*w+x]);
    }
  }
  delete cu_img;
}

TEST(RMDCuTests, deviceImageUploadDownloadFloat2)
{
  cv::Mat img = cv::imread("/home/mpi/Desktop/pict/DSC_0182.JPG", CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat img_flt;
  img.convertTo(img_flt, CV_32F, 1./255.);

  // Opencv gradient computation
  cv::Mat ocv_grad_x, ocv_grad_y;
  cv::Sobel(img_flt, ocv_grad_x, CV_32F, 1, 0, CV_SCHARR);
  cv::Sobel(img_flt, ocv_grad_y, CV_32F, 0, 1, CV_SCHARR);

  const size_t w = img_flt.cols;
  const size_t h = img_flt.rows;

  float2 * cu_grad = new float2[w*h];
  for(size_t y=0; y<h; ++y)
  {
    for(size_t x=0; x<w; ++x)
    {
      cu_grad[y*w+x].x = ocv_grad_x.at<float>(y, x);
      cu_grad[y*w+x].y = ocv_grad_y.at<float>(y, x);
    }
  }

  // upload data to device memory
  rmd::DeviceImage<float2> in_img(w, h);
  in_img.setDevData(cu_grad);

  // download data to host memory
  memset(cu_grad, 0, sizeof(float2)*w*h);
  in_img.getDevData(cu_grad);

  for(size_t y=0; y<h; ++y)
  {
    for(size_t x=0; x<w; ++x)
    {
      ASSERT_FLOAT_EQ(ocv_grad_x.at<float>(y, x), cu_grad[y*w+x].x);
      ASSERT_FLOAT_EQ(ocv_grad_y.at<float>(y, x), cu_grad[y*w+x].y);
    }
  }
  delete cu_grad;
}

TEST(RMDCuTests, deviceImageCopyFloat)
{
  cv::Mat img = cv::imread("/home/mpi/Desktop/pict/DSC_0182.JPG", CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat img_flt;
  img.convertTo(img_flt, CV_32F, 1./255.);

  const size_t w = img_flt.cols;
  const size_t h = img_flt.rows;

  // upload data to gpu memory
  rmd::DeviceImage<float> in_img(w, h);
  in_img.setDevData(reinterpret_cast<float*>(img_flt.data));

  // create copy on device
  rmd::DeviceImage<float> out_img(w, h);
  rmd::copy(in_img, out_img);

  float * cu_img = new float[w*h];
  out_img.getDevData(cu_img);

  for(size_t y=0; y<h; ++y)
  {
    for(size_t x=0; x<w; ++x)
    {
      ASSERT_FLOAT_EQ(img_flt.at<float>(y, x), cu_img[y*w+x]);
    }
  }
  delete cu_img;
}

TEST(RMDCuTests, deviceImageSobelTest)
{
  cv::Mat img = cv::imread("/home/mpi/Desktop/pict/DSC_0182.JPG", CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat img_flt;
  img.convertTo(img_flt, CV_32F, 1./255.);

  // Compare results of the Scharr operator to compute image gradient

  // Opencv gradient computation
  cv::Mat ocv_grad_x, ocv_grad_y;
  cv::Sobel(img_flt, ocv_grad_x, CV_32F, 1, 0, CV_SCHARR);
  cv::Sobel(img_flt, ocv_grad_y, CV_32F, 0, 1, CV_SCHARR);

  // CUDA gradient computation
  const size_t w = img_flt.cols;
  const size_t h = img_flt.rows;

  // upload data to device memory
  rmd::DeviceImage<float> in_img(w, h);
  in_img.setDevData(reinterpret_cast<float*>(img_flt.data));

  // compute gradient on device
  rmd::DeviceImage<float2> out_grad(w, h);
  rmd::sobel(in_img, out_grad);

  // download result to host memory
  float2 * cu_grad = new float2[w*h];
  out_grad.getDevData(cu_grad);

  for(size_t y=1; y<h-1; ++y)
  {
    for(size_t x=1; x<w-1; ++x)
    {
      ASSERT_NEAR(ocv_grad_x.at<float>(y, x), cu_grad[y*w+x].x, 0.00001);
      ASSERT_NEAR(ocv_grad_y.at<float>(y, x), cu_grad[y*w+x].y, 0.00001);
    }
  }
  delete cu_grad;
}
