#include "csr.cuh"
#include "spm_loader.cuh"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

template <typename T> __global__ void spmv_csr_scalar_kernel(CSR_Matrix<T> matrix, const T *dense_vec, T *result) {
    uint row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < matrix.rows) {
        T sum = 0;
        const uint32_t start = matrix.row_ptr[row];
        const uint32_t end = matrix.row_ptr[row + 1];
        for (uint32_t i = start; i < end; i++) {
            sum += matrix.val_p[i] * dense_vec[matrix.col_idx[i]];
        }
        result[row] = sum;
    }
}

template <typename T> __global__ void spmv_csr_vector_kernel(CSR_Matrix<T> matrix, const T *dense_vec, T *result) {
    const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint warp_id = tid >> 5;
    const uint lane = tid & 31;
    uint row = warp_id;

    if (row >= matrix.rows)
        return;
    const uint start = matrix.row_ptr[row];
    const uint end = matrix.row_ptr[row + 1];

    T sum = 0;
    for (int k = start + lane; k < end; k += 32) {
        sum += matrix.val_p[k] * dense_vec[matrix.col_idx[k]];
    }
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);
    }

    if (lane == 0)
        result[row] = sum;
}

template <typename T> bool CSR<T>::load_from_coo(const COO_Matrix<T> &matrix) {
    this->matrix.rows = matrix.rows;
    this->matrix.cols = matrix.cols;
    this->matrix.nnz = matrix.nnz;

    // Histogram of nnz per row, written at index (row+1) so a prefix sum
    // turns it straight into row_ptr.
    this->matrix.row_ptr = static_cast<uint32_t *>(calloc(this->matrix.rows + 1, sizeof(uint32_t)));
    for (uint32_t i = 0; i < matrix.nnz; i++) {
        this->matrix.row_ptr[matrix.row_p[i] + 1]++;
    }
    for (uint32_t i = 0; i < this->matrix.rows; i++) {
        this->matrix.row_ptr[i + 1] += this->matrix.row_ptr[i];
    }

    this->matrix.col_idx = static_cast<uint32_t *>(malloc(this->matrix.nnz * sizeof(uint32_t)));
    this->matrix.val_p = static_cast<T *>(malloc(this->matrix.nnz * sizeof(T)));

    auto offsets = static_cast<uint32_t *>(malloc(this->matrix.rows * sizeof(uint32_t)));
    std::memcpy(offsets, this->matrix.row_ptr, this->matrix.rows * sizeof(uint32_t));

    for (uint32_t i = 0; i < matrix.nnz; i++) {
        const uint32_t row = matrix.row_p[i];
        const uint32_t pos = offsets[row]++;
        this->matrix.col_idx[pos] = matrix.col_p[i];
        this->matrix.val_p[pos] = matrix.val_p[i];
    }

    free(offsets);
    return true;
}

template <typename T> typename CSR<T>::GPU_Pointers CSR<T>::gpu_prep(const T *dense_vec) {
    GPU_Pointers pointers;
    pointers.matrix = this->matrix;

    cudaMalloc(&pointers.matrix.row_ptr, (this->matrix.rows + 1) * sizeof(uint32_t));
    cudaMalloc(&pointers.matrix.col_idx, this->matrix.nnz * sizeof(uint32_t));
    cudaMalloc(&pointers.matrix.val_p, this->matrix.nnz * sizeof(T));

    cudaMalloc(&pointers.dense_vec, this->matrix.cols * sizeof(T));
    cudaMalloc(&pointers.result, this->matrix.rows * sizeof(T));

    cudaMemcpy(pointers.matrix.row_ptr, this->matrix.row_ptr, (this->matrix.rows + 1) * sizeof(uint32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(pointers.matrix.col_idx, this->matrix.col_idx, this->matrix.nnz * sizeof(uint32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(pointers.matrix.val_p, this->matrix.val_p, this->matrix.nnz * sizeof(T), cudaMemcpyHostToDevice);

    cudaMemcpy(pointers.dense_vec, dense_vec, this->matrix.cols * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemset(pointers.result, 0, this->matrix.rows * sizeof(T));
    return pointers;
}

template <typename T> std::vector<T> CSR<T>::gpu_retrive(const GPU_Pointers &pointers) {
    std::vector<T> result(this->matrix.rows);
    cudaMemcpy(result.data(), pointers.result, this->matrix.rows * sizeof(T), cudaMemcpyDeviceToHost);
    return result;
}

template <typename T> void CSR<T>::gpu_free(const GPU_Pointers &pointers) {
    cudaFree(pointers.matrix.row_ptr);
    cudaFree(pointers.matrix.col_idx);
    cudaFree(pointers.matrix.val_p);
    cudaFree(pointers.result);
    cudaFree(pointers.dense_vec);
}

template <typename T> CSR<T>::~CSR() {
    free(this->matrix.val_p);
    free(this->matrix.col_idx);
    free(this->matrix.row_ptr);
}

template class CSR<int>;
template class CSR<float>;
template class CSR<double>;

template __global__ void spmv_csr_scalar_kernel<int>(CSR_Matrix<int>, const int *, int *);
template __global__ void spmv_csr_scalar_kernel<float>(CSR_Matrix<float>, const float *, float *);
template __global__ void spmv_csr_scalar_kernel<double>(CSR_Matrix<double>, const double *, double *);

template __global__ void spmv_csr_vector_kernel<int>(CSR_Matrix<int>, const int *, int *);
template __global__ void spmv_csr_vector_kernel<float>(CSR_Matrix<float>, const float *, float *);
template __global__ void spmv_csr_vector_kernel<double>(CSR_Matrix<double>, const double *, double *);