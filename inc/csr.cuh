#ifndef SPMV_CUDA_CSR_CUH
#define SPMV_CUDA_CSR_CUH
#include "types.h"
#include <sys/types.h>
#include <vector>

template <typename T> __global__ void spmv_csr_scalar_kernel(CSR_Matrix<T> matrix, const T *dense_vec, T *result);
template <typename T> __global__ void spmv_csr_vector_kernel(CSR_Matrix<T> matrix, const T *dense_vec, T *result);

template <typename T> class CSR : public SparseMatrixGPU<T, CSR_Matrix> {
protected:
    using Base = SparseMatrixGPU<T, CSR_Matrix>;
    using GPU_Pointers = typename Base::GPU_Pointers;

  public:
    bool load_from_coo(const COO_Matrix<T> &matrix) override;
    ~CSR() override;
    GPU_Pointers gpu_prep(const T *dense_vec) override;
    std::vector<T> gpu_retrive(const GPU_Pointers &pointers) override;
    void gpu_free(const GPU_Pointers &pointers) override;

};



template <typename T> class CSR_Scalar : public CSR<T> {
    using Base = SparseMatrixGPU<T, CSR_Matrix>;
    using GPU_Pointers = typename Base::GPU_Pointers;
protected:
    void calculate_launch_config() override {
        LaunchConfig cfg{};
        cfg.block_size  = 256;
        cfg.grid_size   = (this->matrix.rows + cfg.block_size - 1) / cfg.block_size;
        cfg.shared_bytes = 0;
        this->launch_config = cfg;
    };

public:
    void gpu_compute(GPU_Pointers *pointers, uint grid_size, uint blk_size) override {
        spmv_csr_scalar_kernel<T><<<grid_size, blk_size>>>(pointers->matrix, pointers->dense_vec, pointers->result);
    }
};

template <typename T> class CSR_Vector : public CSR<T> {
    using Base = SparseMatrixGPU<T, CSR_Matrix>;
    using GPU_Pointers = typename Base::GPU_Pointers;
protected:
    void calculate_launch_config() override {
        LaunchConfig cfg{};
        cfg.block_size  = 128;
        cfg.grid_size   = (this->matrix.rows * 32 + cfg.block_size - 1) / cfg.block_size;
        cfg.shared_bytes = 0;
        this->launch_config = cfg;
    };
public:
    void gpu_compute(GPU_Pointers *pointers, uint grid_size, uint blk_size) override {
        spmv_csr_vector_kernel<T><<<grid_size, blk_size>>>(pointers->matrix, pointers->dense_vec, pointers->result);
    }
};

#endif // SPMV_CUDA_CSR_CUH
