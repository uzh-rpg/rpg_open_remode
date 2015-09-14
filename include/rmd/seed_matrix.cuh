#ifndef SEED_MATRIX_CUH
#define SEED_MATRIX_CUH

#include <cuda_runtime.h>
#include <rmd/device_image.cuh>
#include <rmd/pinhole_camera.cuh>
#include <rmd/device_data.cuh>
#include <rmd/se3.cuh>

namespace rmd
{

namespace ConvergenceStates
{
enum ConvergenceState
{
  UPDATE = 0,
  CONVERGED,
  BORDER,
  DIVERGED,
  NO_MATCH,
  NOT_VISIBLE
};
}
typedef ConvergenceStates::ConvergenceState ConvergenceState;

class SeedMatrix
{
public:
  SeedMatrix(
      const size_t &width,
      const size_t &height,
      const PinholeCamera &cam);
  bool setReferenceImage(
      float *host_ref_img_align_row_maj,
      const SE3<float> &T_curr_world,
      const float &min_depth,
      const float &max_depth);
  bool update(
      float *host_curr_img_align_row_maj,
      const SE3<float> &T_curr_world);
  void downloadDepthmap(float *host_depthmap_align_row_maj) const;

  const DeviceImage<float> & getMu() const;
  const DeviceImage<float> & getSigmaSq() const;
  const DeviceImage<float> & getA() const;
  const DeviceImage<float> & getB() const;

#if RMD_DEBUG
  void downloadSigmaSq(float *host_align_row_maj) const;
  void downloadA(float *host_align_row_maj) const;
  void downloadB(float *host_align_row_maj) const;
  void downloadSumTempl(float *host_align_row_maj) const;
  void downloadConstTemplDenom(float *host_align_row_maj) const;
  void downloadConvergence(int *host_align_row_maj) const;
  void downloadEpipolarMatches(float2 *host_align_row_maj) const;
  int getPatchSide() const { return dev_data_.patch.side; }
#endif

private:
  size_t width_;
  size_t height_;
  DeviceImage<float> ref_img_, curr_img_;
  // Template statistics for NCC (pre)computation
  DeviceImage<float> sum_templ_, const_templ_denom_;
  // Measurement parameters
  DeviceImage<float> mu_, sigma_, a_, b_;
  // Convergence state
  DeviceImage<int> convergence_;
  // Epipolar matches
  DeviceImage<float2> epipolar_matches_;
  DeviceData dev_data_;
  SE3<float> T_world_ref_;
  // kernel config
  dim3 dim_block_;
  dim3 dim_grid_;
};

} // rmd namespace

#endif // SEED_MATRIX_CUH
