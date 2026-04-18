#ifndef SPMV_CUDA_SPM_LOADER_CUH
#define SPMV_CUDA_SPM_LOADER_CUH

#include "coo.cuh"
#include <string>
template <typename T>
class MatrixMarketLoader {
public:
    static bool load(const std::string &path, typename COO<T>::COO_Matrix &out);
};

#endif // SPMV_CUDA_SPM_LOADER_CUH
