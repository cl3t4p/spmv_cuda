#ifndef SPMV_CUDA_CUSPARSE_SPMV_CUH
#define SPMV_CUDA_CUSPARSE_SPMV_CUH

#include "coo.cuh"
#include "csr.cuh"
#include <cusparse.h>
#include <type_traits>


template <typename T> constexpr cudaDataType cusparse_value_type() {
    if constexpr (std::is_same_v<T, float>) {
        return CUDA_R_32F;
    }else{
        return CUDA_R_32I;
    }
}

template <typename T> class CSR_CuSparse : public CSR<T> {
    using Base = SparseMatrixGPU<T, CSR_Matrix>;
    using GPU_Pointers = typename Base::GPU_Pointers;

  protected:
    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t mat_desc = nullptr;
    cusparseDnVecDescr_t vec_x = nullptr;
    cusparseDnVecDescr_t vec_y = nullptr;
    void *d_buffer = nullptr;
    size_t buffer_size = 0;

    void calculate_launch_config() override {
        LaunchConfig cfg{};
        cfg.block_size = 1;
        cfg.grid_size = 1;
        cfg.shared_bytes = 0;
        this->launch_config = cfg;
    }

  public:
    GPU_Pointers gpu_prep(const T *dense_vec) override;
    void gpu_compute(GPU_Pointers *pointers, uint grid_size, uint blk_size) override;
    void gpu_free(const GPU_Pointers &pointers) override;
};

template <typename T> class COO_CuSparse : public COO<T> {
    using Base = SparseMatrixGPU<T, COO_Matrix>;
    using GPU_Pointers = typename Base::GPU_Pointers;

  protected:
    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t mat_desc = nullptr;
    cusparseDnVecDescr_t vec_x = nullptr;
    cusparseDnVecDescr_t vec_y = nullptr;
    void *d_buffer = nullptr;
    size_t buffer_size = 0;

    void calculate_launch_config() override {
        LaunchConfig cfg{};
        cfg.block_size = 1;
        cfg.grid_size = 1;
        cfg.shared_bytes = 0;
        this->launch_config = cfg;
    }

  public:
    bool load_from_coo(const COO_Matrix<T> &matrix) override;
    GPU_Pointers gpu_prep(const T *dense_vec) override;
    void gpu_compute(GPU_Pointers *pointers, uint grid_size, uint blk_size) override;
    void gpu_free(const GPU_Pointers &pointers) override;
};

#endif // SPMV_CUDA_CUSPARSE_SPMV_CUH
