#include <gtest/gtest.h>
#include <cuda_runtime_api.h>

/// Run all the tests that were declared with TEST()
int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();

  cudaDeviceReset();

  return ret;
}
