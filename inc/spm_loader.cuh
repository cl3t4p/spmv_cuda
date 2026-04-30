#ifndef SPMV_CUDA_SPM_LOADER_CUH
#define SPMV_CUDA_SPM_LOADER_CUH

#include "types.cuh"
#include <string>

template <typename T> class MatrixMarketLoader {
  public:
    static bool load(const std::string &path, COO_Matrix<T> &out);
    static void free_matrix(const COO_Matrix<T> &matrix);
};

#endif // SPMV_CUDA_SPM_LOADER_CUH
