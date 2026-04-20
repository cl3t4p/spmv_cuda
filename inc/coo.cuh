#ifndef SPMV_MATRIX_MARKET_H
#define SPMV_MATRIX_MARKET_H
#include "types.h"
#include <cstdint>

template <typename T> __global__ void spmv_coo_kernel(COO_Matrix<T> matrix, const T *dense_vec, T *result);

template <typename T> class COO : public SparseMatrixGPU<T, COO_Matrix> {
    using Base = SparseMatrixGPU<T, COO_Matrix>;
    using GPU_Pointers = typename Base::GPU_Pointers;

  public:
    bool load_from_coo(const COO_Matrix<T> &matrix) override;
    ~COO() override;
    GPU_Pointers gpu_prep(const T *dense_vec) const override;
    void gpu_free(const GPU_Pointers &pointers) override;
    std::vector<T> gpu_retrive(const GPU_Pointers &pointers) override;

    __forceinline__ void gpu_compute(GPU_Pointers *pointers, uint grid_size, uint blk_size) override {
        spmv_coo_kernel<T><<<grid_size, blk_size>>>(pointers->matrix, pointers->dense_vec, pointers->result);
    }
};

#endif // SPMV_MATRIX_MARKET_H
