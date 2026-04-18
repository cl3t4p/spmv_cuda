#include "coo.cuh"
#include "spm_loader.cuh"
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

template <typename T>
__global__ void spmv_coo_kernel(typename COO<T>::COO_Matrix matrix,
                                const T *dense_vec, T *result) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < matrix.nnz) {
    const uint row = matrix.row_p[i];
    const uint col = matrix.col_p[i];
    const T val = matrix.val_p[i];
    atomicAdd(&result[row], val * dense_vec[col]);
  }
}

template <typename T>
void COO<T>::gpu_compute(GPU_COO_Pointers *pointers, uint grid_size,
                         uint blk_size) {
  spmv_coo_kernel<T><<<grid_size, blk_size>>>(
      pointers->matrix, pointers->dense_vec, pointers->result);
}

template <typename T>
bool COO<T>::load_from_file(const std::string &path) {
  return MatrixMarketLoader<T>::load(path, matrix);
}

template <typename T>
std::vector<T> COO<T>::cpu_compute(const std::vector<T> &dense_vec) const {
  std::vector<T> result(getCols(), 0);

#pragma omp parallel for schedule(static)
  for (uint32_t i = 0; i < matrix.nnz; i++) {
#pragma omp atomic
    result[matrix.row_p[i]] +=
        matrix.val_p[i] * dense_vec[matrix.col_p[i]];
  }

  return result;
}

template <typename T>
typename COO<T>::GPU_COO_Pointers COO<T>::gpu_prep(const T *dense_vec) const {
  const uint32_t nnz = matrix.nnz;
  GPU_COO_Pointers pointers;

  pointers.matrix = matrix;

  cudaMalloc(&pointers.matrix.row_p, nnz * sizeof(uint32_t));
  cudaMalloc(&pointers.matrix.col_p, nnz * sizeof(uint32_t));
  cudaMalloc(&pointers.matrix.val_p, nnz * sizeof(T));

  cudaMalloc(&pointers.dense_vec, getCols() * sizeof(T));
  cudaMalloc(&pointers.result, getCols() * sizeof(T));

  cudaMemcpy(pointers.matrix.row_p, matrix.row_p, nnz * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(pointers.matrix.col_p, matrix.col_p, nnz * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(pointers.matrix.val_p, matrix.val_p, nnz * sizeof(T),
             cudaMemcpyHostToDevice);

  cudaMemcpy(pointers.dense_vec, dense_vec, getCols() * sizeof(T),
             cudaMemcpyHostToDevice);
  cudaMemset(pointers.result, 0, getCols() * sizeof(T));
  return pointers;
}

template <typename T>
std::vector<T>
COO<T>::gpu_retrive(const GPU_COO_Pointers &pointers) const {
  std::vector<T> result(getCols());
  cudaMemcpy(result.data(), pointers.result, getCols() * sizeof(T),
             cudaMemcpyDeviceToHost);
  return result;
}

template <typename T>
void COO<T>::gpu_free(const GPU_COO_Pointers &pointers) {
  cudaFree(pointers.matrix.row_p);
  cudaFree(pointers.matrix.col_p);
  cudaFree(pointers.matrix.val_p);
  cudaFree(pointers.result);
  cudaFree(pointers.dense_vec);
}

template <typename T> COO<T>::~COO() {
  free(matrix.val_p);
  free(matrix.col_p);
  free(matrix.row_p);
}

// Explicit instantiations. atomicAdd supports int, float, double (sm_60+).
template class COO<int>;
template class COO<float>;
template class COO<double>;

template __global__ void
spmv_coo_kernel<int>(COO<int>::COO_Matrix, const int *, int *);
template __global__ void
spmv_coo_kernel<float>(COO<float>::COO_Matrix, const float *, float *);
template __global__ void
spmv_coo_kernel<double>(COO<double>::COO_Matrix, const double *,
                        double *);
