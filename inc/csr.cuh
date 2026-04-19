#ifndef SPMV_CSR_CUH
#define SPMV_CSR_CUH
#include <cstdint>
#include <string>
#include <sys/types.h>
#include <vector>
#include "types.h"

template <typename T> class CSR : public SparseMatrixGPU<T,CSR_Matrix>{
  using Base = SparseMatrixGPU<T, CSR_Matrix>;
  using GPU_Pointers = typename Base::GPU_Pointers;
public:
bool load_from_file(const std::string &path) override;
  ~CSR() override;
  GPU_Pointers gpu_prep(const T *dense_vec) const override;
  void gpu_free(const GPU_Pointers &pointers) override;
  void gpu_compute(GPU_Pointers *pointers, uint grid_size,
                   uint blk_size) override;
  std::vector<T> gpu_retrive(const GPU_Pointers &pointers) override;
};

template <typename T>
__global__ void spmv_csr_kernel(CSR_Matrix<T> matrix,
                                const T *dense_vec, T *result);

#endif // SPMV_CSR_CUH
