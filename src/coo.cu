#include "coo.cuh"
#include "spm_loader.cuh"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

template <typename T> __global__ void spmv_coo_kernel(COO_Matrix<T> matrix, const T *dense_vec, T *result) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < matrix.nnz) {
        const uint row = matrix.row_p[i];
        const uint col = matrix.col_p[i];
        const T val = matrix.val_p[i];
        atomicAdd(&result[row], val * dense_vec[col]);
    }
}

template <typename T> bool COO<T>::load_from_coo(const COO_Matrix<T> &matrix) {
    this->matrix.nnz = matrix.nnz;
    this->matrix.rows = matrix.rows;
    this->matrix.cols = matrix.cols;

    this->matrix.row_p = static_cast<uint32_t *>(malloc(sizeof(uint32_t) * matrix.nnz));
    this->matrix.col_p = static_cast<uint32_t *>(malloc(sizeof(uint32_t) * matrix.nnz));
    this->matrix.val_p = static_cast<T *>(malloc(sizeof(T) * matrix.nnz));

    if (this->matrix.row_p == nullptr || this->matrix.col_p == nullptr || this->matrix.row_p == nullptr) {
        std::cerr << "error in matrix malloc " << std::endl;
        return false;
    }

    std::memcpy(this->matrix.row_p, matrix.row_p, sizeof(uint32_t) * matrix.nnz);
    std::memcpy(this->matrix.col_p, matrix.col_p, sizeof(uint32_t) * matrix.nnz);
    std::memcpy(this->matrix.val_p, matrix.val_p, sizeof(T) * matrix.nnz);
    return true;
}

template <typename T> typename COO<T>::GPU_Pointers COO<T>::gpu_prep(const T *dense_vec) {
    const uint32_t nnz = this->matrix.nnz;
    GPU_Pointers pointers;

    pointers.matrix = this->matrix;

    cudaMalloc(&pointers.matrix.row_p, nnz * sizeof(uint32_t));
    cudaMalloc(&pointers.matrix.col_p, nnz * sizeof(uint32_t));
    cudaMalloc(&pointers.matrix.val_p, nnz * sizeof(T));

    cudaMalloc(&pointers.dense_vec, this->getCols() * sizeof(T));
    cudaMalloc(&pointers.result, this->getRows() * sizeof(T));

    cudaMemcpy(pointers.matrix.row_p, this->matrix.row_p, nnz * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(pointers.matrix.col_p, this->matrix.col_p, nnz * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(pointers.matrix.val_p, this->matrix.val_p, nnz * sizeof(T), cudaMemcpyHostToDevice);

    cudaMemcpy(pointers.dense_vec, dense_vec, this->getCols() * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemset(pointers.result, 0, this->getRows() * sizeof(T));
    return pointers;
}

template <typename T> void COO<T>::gpu_free(const GPU_Pointers &pointers) {
    cudaFree(pointers.matrix.row_p);
    cudaFree(pointers.matrix.col_p);
    cudaFree(pointers.matrix.val_p);
    cudaFree(pointers.result);
    cudaFree(pointers.dense_vec);
}

template <typename T> COO<T>::~COO() {
    free(this->matrix.val_p);
    free(this->matrix.col_p);
    free(this->matrix.row_p);
}

// Explicit instantiations. atomicAdd supports int, float, double (sm_60+).
template class COO<int>;
template class COO<float>;
template class COO<double>;

template __global__ void spmv_coo_kernel<int>(COO_Matrix<int>, const int *, int *);
template __global__ void spmv_coo_kernel<float>(COO_Matrix<float>, const float *, float *);
template __global__ void spmv_coo_kernel<double>(COO_Matrix<double>, const double *, double *);
