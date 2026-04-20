#ifndef SPMV_CUDA_COO_CUSPARSE_CUH
#define SPMV_CUDA_COO_CUSPARSE_CUH
#include "types.h"
#include <cstdint>
#include <cusparse.h>

template <typename T> class COO_Cusparse : public SparseMatrixGPU<T, COO_Matrix> {
    using Base = SparseMatrixGPU<T, COO_Matrix>;
    using GPU_Pointers = typename Base::GPU_Pointers;

    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t mat_desc = nullptr;
    cusparseDnVecDescr_t vec_x = nullptr;
    cusparseDnVecDescr_t vec_y = nullptr;
    void *buffer = nullptr;
    size_t buffer_size = 0;

  public:
    bool load_from_coo(const COO_Matrix<T> &matrix) override;
    ~COO_Cusparse() override;
    GPU_Pointers gpu_prep(const T *dense_vec) override;
    void gpu_free(const GPU_Pointers &pointers) override;
    std::vector<T> gpu_retrive(const GPU_Pointers &pointers) override;

    __forceinline__ void gpu_compute(GPU_Pointers *pointers, uint grid_size, uint blk_size) override {
        (void)pointers;
        (void)grid_size;
        (void)blk_size;
        const T alpha = static_cast<T>(1);
        const T beta = static_cast<T>(0);
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_desc, vec_x, &beta, vec_y,
                     std::is_same_v<T, float> ? CUDA_R_32F : CUDA_R_64F, CUSPARSE_SPMV_COO_ALG1, buffer);
    }
};

#endif // SPMV_CUDA_COO_CUSPARSE_CUH
