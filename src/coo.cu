#include "coo.cuh"
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <vector>

__global__ void spmv_coo_kernel(const COO_Matrix &matrix,
                                const double *dense_vec, double *result) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < matrix.nnz) {
    const uint row = matrix.row_p[i];
    const uint col = matrix.col_p[i];
    const double val = matrix.val_p[i];
    atomicAdd(&result[row], val * dense_vec[col]);
  }
}

std::vector<std::string> split_line(const std::string &line) {
  std::istringstream iss(line);
  std::vector<std::string> result;
  std::string val;
  while (iss >> val) {
    result.push_back(val);
  }

  return result;
}

bool COO::load_from_file(const std::string &path) {
  auto ifs_mtx = std::ifstream(path);
  if (!ifs_mtx.is_open()) {
    std::cerr << "file " << path << " does not exists!" << std::endl;
    return false;
  }
  std::string line;
  while (std::getline(ifs_mtx, line)) {
    if (line.at(0) != '%') {
      break;
    }
  }

  auto coo_info = split_line(line);
  matrix.rows = std::stoull(coo_info[0]);
  matrix.cols = std::stoull(coo_info[1]);
  matrix.nnz = std::stoull(coo_info[2]);

  // Malloc
  matrix.row_p = static_cast<uint32_t *>(malloc(sizeof(uint32_t) * matrix.nnz));
  matrix.col_p = static_cast<uint32_t *>(malloc(sizeof(uint32_t) * matrix.nnz));
  matrix.val_p = static_cast<double *>(malloc(sizeof(double) * matrix.nnz));

  int index = 0;

  while (std::getline(ifs_mtx, line)) {
    auto entry_info = split_line(line);
    matrix.row_p[index] = std::stoull(entry_info[0]);
    matrix.col_p[index] = std::stoull(entry_info[1]);
    matrix.val_p[index] = std::stod(entry_info[2]);
    index++;
  }

  if (matrix.nnz != index) {
    std::cerr << "nnz " << matrix.nnz << " size does not match matrix" << index
              << "!" << std::endl;
    return false;
  }
  return true;
}

std::vector<double> COO::cpu_compute(const std::vector<double> &dense_vec) {
  std::vector<double> result(getCols(), 0.0);

#pragma omp parallel for schedule(static)
  for (uint32_t i = 0; i < matrix.nnz; i++) {
#pragma omp atomic
    result[matrix.row_p[i] - 1] +=
        matrix.val_p[i] * dense_vec[matrix.col_p[i] - 1];
  }

  return result;
}

GPU_COO_Pointers COO::gpu_prep(const double *dense_vec) {
  const uint32_t nnz = matrix.nnz;
  GPU_COO_Pointers pointers;

  pointers.matrix = matrix;

  // Memory GPU
  //  Pointer for the Sparse Matrix stored in the COO_Matrix
  cudaMalloc(&pointers.matrix.row_p, nnz * sizeof(uint32_t));
  cudaMalloc(&pointers.matrix.col_p, nnz * sizeof(uint32_t));
  cudaMalloc(&pointers.matrix.val_p, nnz * sizeof(double));

  cudaMalloc(&pointers.dense_vec, getCols() * sizeof(double));
  cudaMalloc(&pointers.result, getCols() * sizeof(double));

  // Copy
  cudaMemcpy(pointers.matrix.row_p, matrix.row_p, nnz * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(pointers.matrix.col_p, matrix.col_p, nnz * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(pointers.matrix.val_p, matrix.val_p, nnz * sizeof(double),
             cudaMemcpyHostToDevice);

  cudaMemcpy(pointers.dense_vec, dense_vec, getCols() * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemset(pointers.result, 0, getCols() * sizeof(double));
  return pointers;
}

std::vector<double> COO::gpu_retrive(const GPU_COO_Pointers &pointers) {
  std::vector<double> result(getCols());
  cudaMemcpy(result.data(), pointers.result, getCols() * sizeof(double),
             cudaMemcpyDeviceToHost);
  return result;
}

void COO::gpu_free(const GPU_COO_Pointers &pointers) {
  cudaFree(pointers.matrix.row_p);
  cudaFree(pointers.matrix.col_p);
  cudaFree(pointers.matrix.val_p);
  cudaFree(pointers.result);
  cudaFree(pointers.dense_vec);
}

COO::~COO() {
  free(matrix.val_p);
  free(matrix.col_p);
  free(matrix.row_p);
}
