#include <cuda_runtime_api.h>
#include <iostream>
#include <ostream>
#include <string>
#include <sys/time.h>
#include <vector>

#include "coo.cuh"
#include "utils.h"

template <typename T> int run(const char *path) {
  COO<T> coo;

  std::cout << "loading matrix" << std::endl;
  if (!coo.load_from_file(path)) {
    return -1;
  } else {
    std::cout << "matrix loaded with success!" << std::endl;
  }

  std::vector<T> dense_vec(coo.getRows(), static_cast<T>(1));

  TIMER_DEF;
  std::vector<float> gpu_times;
  float cputime, gputime;
  double error;

  std::cout << "CPU : Compute" << std::endl;
  TIMER_START;
  auto cpu_result = coo.cpu_compute(dense_vec);
  TIMER_STOP;

  cputime = TIMER_ELAPSED;

  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, 0);
  uint blk_size = std::min(128, props.maxThreadsPerBlock);
  uint grd_size = (coo.getMatrix().nnz + blk_size - 1) / blk_size;

  auto gpu_pointers = coo.gpu_prep(dense_vec.data());
  std::cout << "GPU : Compute" << std::endl;

  // warmup
  for (int x = 0; x < 10; x++) {
    coo.free_result(&gpu_pointers);
    coo.gpu_compute(&gpu_pointers, grd_size, blk_size);
    cudaDeviceSynchronize();
  }

  // benchmark
  for (int x = 0; x < 100; x++) {
    coo.free_result(&gpu_pointers);

    TIMER_START;
    coo.gpu_compute(&gpu_pointers, grd_size, blk_size);
    cudaDeviceSynchronize();
    TIMER_STOP;
    gpu_times.push_back(TIMER_ELAPSED);
  }
  auto gpu_result = coo.gpu_retrive(gpu_pointers);
  COO<T>::gpu_free(gpu_pointers);

  std::cout << "Comparing" << std::endl;

  bool result = compare_vectors<T>(cpu_result, gpu_result);
  error = diff_vector<T>(cpu_result, gpu_result);

  if (result) {
    std::cout << "2 Vector are similar" << std::endl;
  } else {
    std::cout << "2 Vector not are similar!!!" << std::endl;
  }

  gputime = 0;
  for (auto time : gpu_times) {
    gputime += time;
  }
  gputime = gputime / gpu_times.size();

  printf("================================== Times and results of my code "
         "==================================\n");
  printf("Error between CPU and GPU is %lf\n", error);
  printf("\nVector len = %d, CPU time = %5.3f\n", coo.getMatrix().nnz, cputime);
  printf("\nblk_size = %d, grd_size = %d, GPU time (gettimeofday): %5.3f sec\n",
         blk_size, grd_size, gputime);
  return 0;
}



int main(int argc, char **argv) {
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " <dtype> <file>" << std::endl;
    std::cout << "  dtype: int | float | double" << std::endl;
    return 1;
  }

  const std::string dtype = argv[1];
  const char *path = argv[2];

  if (dtype == "int") {
    return run<int>(path);
  } else if (dtype == "float") {
    return run<float>(path);
  } else if (dtype == "double") {
    return run<double>(path);
  } else {
    std::cerr << "Unknown dtype: " << dtype
              << " (expected int, float, or double)" << std::endl;
    return 1;
  }
}
