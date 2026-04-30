#ifndef SPMV_CUDA_COO_CUH
#define SPMV_CUDA_COO_CUH
#include "types.cuh"
#include <cstdint>

template <typename T> __global__ void spmv_coo_kernel(COO_Matrix<T> matrix, const T *dense_vec, T *result);
template <typename T> __global__ void spmv_coo_optimized_kernel(COO_Matrix<T> matrix,const T *dense_vec, T *result);

template <typename T> class COO : public SparseMatrixGPU<T, COO_Matrix> {
    using Base = SparseMatrixGPU<T, COO_Matrix>;
    using GPU_Pointers = typename Base::GPU_Pointers;

  protected:
    void calculate_launch_config() override {
        LaunchConfig cfg{};
        cfg.block_size = 256;
        cfg.grid_size = (this->matrix.nnz + cfg.block_size - 1) / cfg.block_size;
        cfg.shared_bytes = 0;
        this->launch_config = cfg;
    };

  public:
    bool load_from_coo(const COO_Matrix<T> &matrix) override;
    ~COO() override;
    GPU_Pointers gpu_prep(const T *dense_vec) override;
    void gpu_free(const GPU_Pointers &pointers) override;

    void gpu_compute(GPU_Pointers *pointers, uint grid_size, uint blk_size) override {
        spmv_coo_kernel<T><<<grid_size, blk_size>>>(pointers->matrix, pointers->dense_vec, pointers->result);
    }
};



template <typename T> class COO_Optimized : public COO<T> {
    using Base = SparseMatrixGPU<T, COO_Matrix>;
    using GPU_Pointers = typename Base::GPU_Pointers;
public:
    void gpu_compute(GPU_Pointers *pointers, uint grid_size, uint blk_size) override {
        spmv_coo_optimized_kernel<T><<<grid_size, blk_size>>>(pointers->matrix, pointers->dense_vec, pointers->result);
    }


};
#endif // SPMV_CUDA_COO_CUH
