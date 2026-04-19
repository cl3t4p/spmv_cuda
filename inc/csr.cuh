#ifndef SPMV_CSR_CUH
#define SPMV_CSR_CUH
#include "types.h"
#include <string>
#include <sys/types.h>
#include <vector>

template <typename T> class CSR : public SparseMatrixGPU<T, CSR_Matrix> {
    using Base = SparseMatrixGPU<T, CSR_Matrix>;
    using GPU_Pointers = typename Base::GPU_Pointers;

  private:
    COO_Matrix<T> coo_matrix;

  public:
    bool load_from_file(const std::string &path) override;
    ~CSR() override;
    GPU_Pointers gpu_prep(const T *dense_vec) const override;

    std::vector<T> gpu_retrive(const GPU_Pointers &pointers) override;

    void gpu_free(const GPU_Pointers &pointers) override;
    void gpu_compute(GPU_Pointers *pointers, uint grid_size, uint blk_size) override;

    COO_Matrix<T> get_coo_matrix() override { return this->coo_matrix; }
};

template <typename T> __global__ void spmv_csr_kernel(CSR_Matrix<T> matrix, const T *dense_vec, T *result);

#endif // SPMV_CSR_CUH
