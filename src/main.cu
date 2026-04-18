#include <cuda_runtime_api.h>
#include <iostream>
#include <ostream>
#include <sys/time.h>
#include <vector>

#include "coo.cuh"

#define TIMER_DEF struct timeval temp_1, temp_2

#define TIMER_START gettimeofday(&temp_1, (struct timezone *)0)

#define TIMER_STOP gettimeofday(&temp_2, (struct timezone *)0)

#define TIMER_ELAPSED                                                          \
  ((temp_2.tv_sec - temp_1.tv_sec) +                                           \
   (temp_2.tv_usec - temp_1.tv_usec) / 1000000.0)

typedef double dtype;

bool compare_vectors(const std::vector<dtype> &a, const std::vector<dtype> &b,
                     double eps = 1e-9) {
  if (a.size() != b.size()) {
    return false;
  }

  for (size_t i = 0; i < a.size(); ++i) {
    if (std::abs(a[i] - b[i]) > eps) {
      return false;
    }
  }
  return true;
}

dtype diff_vector(const std::vector<dtype> &a, const std::vector<dtype> &b) {
  if (a.size() != b.size()) {
    return -1;
  }
  dtype result = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    result += std::abs(a[i] - b[i]);
  }
  return result / static_cast<dtype>(a.size());
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <file>" << std::endl;
    return 1;
  }

  COO coo;

  std::cout << "loading matrix" << std::endl;
  if (!coo.load_from_file(argv[1])) {
    return -1;
  } else {
    std::cout << "matrix loaded with success!" << std::endl;
  }

  std::vector<double> dense_vec(coo.getRows(), 1.0);

  TIMER_DEF;
  std::vector<float> gpu_times;
  float error, cputime, gputime;

  std::cout << "CPU : Compute" << std::endl;
  TIMER_START;
  auto cpu_result = coo.cpu_compute(dense_vec);
  TIMER_STOP;

  cputime = TIMER_ELAPSED;

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  int blk_size = std::min(128, props.maxThreadsPerBlock);
  int grd_size = (coo.getMatrix().nnz + blk_size - 1) / blk_size;

  GPU_COO_Pointers gpu_pointers = coo.gpu_prep(dense_vec.data());
  std::cout << "GPU : Compute" << std::endl;

  // warmup
  for (int x = 0; x < 10; x++) {
    coo.free_result(&gpu_pointers); // reset output
    coo.gpu_compute(&gpu_pointers, grd_size, blk_size);
    cudaDeviceSynchronize();
  }

  // benchmark
  for (int x = 0; x < 100; x++) {
    coo.free_result(&gpu_pointers); // reset output

    TIMER_START;
    coo.gpu_compute(&gpu_pointers, grd_size, blk_size);
    cudaDeviceSynchronize();
    TIMER_STOP;
    gpu_times.push_back(TIMER_ELAPSED);
  }
  auto gpu_result = coo.gpu_retrive(gpu_pointers);
  coo.gpu_free(gpu_pointers);

  std::cout << "Comparing" << std::endl;

  bool result = compare_vectors(cpu_result, gpu_result);
  error = diff_vector(cpu_result, gpu_result);

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

  // ============================================ Print the results
  // ============================================

  printf("================================== Times and results of my code "
         "==================================\n");
  printf("Error between CPU and GPU is %lf\n", error);
  printf("\nVector len = %d, CPU time = %5.3f\n", coo.getMatrix().nnz, cputime);
  printf("\nblk_size = %d, grd_size = %d, GPU time (gettimeofday): %5.3f sec\n",
         blk_size, grd_size, gputime);
}
