#include <cuda_runtime_api.h>
#include <iostream>
#include <ostream>
#include <string>
#include <sys/time.h>
#include <vector>


#include "coo.cuh"
#include "utils.h"


template <typename Matrix, typename T>
int run(const char *path) {
  Matrix mat;


  std::cout << "loading matrix" << std::endl;
  if (!mat.load_from_file(path)) {
    return -1;
  } else {
    std::cout << "matrix loaded with success!" << std::endl;
  }

  std::vector<T> dense_vec(mat.getRows(), static_cast<T>(1));

  TIMER_DEF;
  std::vector<float> gpu_times;
  float cputime, gputime;
  double error;

  std::cout << "CPU : Compute" << std::endl;
  TIMER_START;
  auto cpu_result = cpu_compute<T>(mat.getMatrix(),dense_vec);
  TIMER_STOP;

  cputime = TIMER_ELAPSED;

  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, 0);
  uint blk_size = std::min(128, props.maxThreadsPerBlock);
  uint grd_size = (mat.getMatrix().nnz + blk_size - 1) / blk_size;

  auto gpu_pointers = mat.gpu_prep(dense_vec.data());
  std::cout << "GPU : Compute" << std::endl;

  // warmup
  for (int x = 0; x < 10; x++) {
    mat.gpu_free_result(&gpu_pointers);
    mat.gpu_compute(&gpu_pointers, grd_size, blk_size);
    cudaDeviceSynchronize();
  }

  // benchmark
  for (int x = 0; x < 100; x++) {
    mat.gpu_free_result(&gpu_pointers);

    TIMER_START;
    mat.gpu_compute(&gpu_pointers, grd_size, blk_size);
    cudaDeviceSynchronize();
    TIMER_STOP;
    gpu_times.push_back(TIMER_ELAPSED);
  }
  auto gpu_result = mat.gpu_retrive(gpu_pointers);
  mat.gpu_free(gpu_pointers);

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
  printf("\nVector len = %d, CPU time = %5.3f\n", mat.getMatrix().nnz, cputime);
  printf("\nblk_size = %d, grd_size = %d, GPU time (gettimeofday): %5.3f sec\n",
         blk_size, grd_size, gputime);
  return 0;
}


template <typename T>
int run_by_format(const std::string& matrix_type, const char* path) {
  if (matrix_type == "coo") {
    return run<COO<T>, T>(path);
  }
  if (matrix_type == "csr") {
    std::cerr << "CSR not implemented yet" << std::endl;
    return 1;
  }
  std::cerr << "Unknown matrix type: " << matrix_type
      << " (expected coo or csr)" << std::endl;
  return 1;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cout << "Usage: " << argv[0] << " <dtype> <matrix_type> <file>" << std::endl;
    std::cout << "  dtype: int | float | double" << std::endl;
    std::cout << "  matrix_type: coo | csr" << std::endl;
    return 1;
  }

  const std::string dtype = argv[1];
  const std::string matrix_type = argv[2];
  const char* path = argv[3];

  if (dtype == "int") {
    return run_by_format<int>(matrix_type, path);
  } else if (dtype == "float") {
    return run_by_format<float>(matrix_type, path);
  } else if (dtype == "double") {
    return run_by_format<double>(matrix_type, path);
  } else {
    std::cerr << "Unknown dtype: " << dtype
              << " (expected int, float, or double)" << std::endl;
    return 1;
  }
}
