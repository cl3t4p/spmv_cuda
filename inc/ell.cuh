#ifndef SPMV_CUDA_ELL_CUH
#define SPMV_CUDA_ELL_CUH
#include "types.cuh"

template <typename T> struct ELL_Matrix : BASE_Matrix {
    uint max_col_len;
    uint *col_idx;
    T *values;
};

template <typename T> __global__ void spmv_ell_kernel(ELL_Matrix<T> matrix, const T *dense_vec, T *result);

template <typename T> class ELL : public SparseMatrixGPU<T, ELL_Matrix> {
    using Base = SparseMatrixGPU<T, ELL_Matrix>;
    using GPU_Pointers = typename Base::GPU_Pointers;

  protected:
    void calculate_launch_config() override {
        LaunchConfig cfg{};
        cfg.block_size = 256;
        cfg.grid_size = (this->matrix.rows + cfg.block_size - 1) / cfg.block_size;
        cfg.shared_bytes = 0;
        this->launch_config = cfg;
    };

  public:
    bool load_from_coo(const COO_Matrix<T> &og_matrix) override;
    ~ELL() override;
    GPU_Pointers gpu_prep(const T *dense_vec) override;
    void gpu_free(const GPU_Pointers &pointers) override;

    void gpu_compute(GPU_Pointers *pointers, uint grid_size, uint blk_size) override {
        spmv_ell_kernel<T><<<grid_size, blk_size>>>(pointers->matrix, pointers->dense_vec, pointers->result);
    }
};
#endif // SPMV_CUDA_ELL_CUH
