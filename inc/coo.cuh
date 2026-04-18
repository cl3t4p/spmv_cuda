#ifndef SPMV_MATRIX_MARKET_H
#define SPMV_MATRIX_MARKET_H
#include <cstdint>
#include "types.h"


template <typename T>
class COO : public SparseMatrixGPU<T,COO_Matrix>{
  using Base = SparseMatrixGPU<T, COO_Matrix>;
  using GPU_Pointers = typename Base::GPU_Pointers;
public:

  bool load_from_file(const std::string &path) override;
  ~COO();

  GPU_Pointers gpu_prep(const T *dense_vec) const override;
  void gpu_free(const GPU_Pointers &pointers) override;
  void gpu_compute(GPU_Pointers *pointers, uint grid_size, uint blk_size) override;
  std::vector<T> gpu_retrive(const GPU_Pointers &pointers) override;
  void gpu_free_result(GPU_Pointers *pointers) override {
    cudaMemset(pointers->result, 0, Base::getRows() * sizeof(T));
  }

};

template <typename T>
__global__ void spmv_coo_kernel(COO_Matrix<T> matrix,
                                const T *dense_vec, T *result);

#endif // SPMV_MATRIX_MARKET_H
