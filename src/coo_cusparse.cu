#include "coo_cusparse.cuh"
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <vector>

template <typename T> static constexpr cudaDataType cusparse_dtype() {
    if constexpr (std::is_same_v<T, float>) {
        return CUDA_R_32F;
    } else {
        return CUDA_R_64F;
    }
}

template <typename T> bool COO_Cusparse<T>::load_from_coo(const COO_Matrix<T> &matrix) {
    this->matrix.nnz = matrix.nnz;
    this->matrix.rows = matrix.rows;
    this->matrix.cols = matrix.cols;

    this->matrix.row_p = static_cast<uint32_t *>(malloc(sizeof(uint32_t) * matrix.nnz));
    this->matrix.col_p = static_cast<uint32_t *>(malloc(sizeof(uint32_t) * matrix.nnz));
    this->matrix.val_p = static_cast<T *>(malloc(sizeof(T) * matrix.nnz));

    if (!this->matrix.row_p || !this->matrix.col_p || !this->matrix.val_p) {
        std::cerr << "error in matrix malloc" << std::endl;
        return false;
    }

    // cuSPARSE SpMV requires COO entries sorted by row.
    std::vector<uint32_t> perm(matrix.nnz);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](uint32_t a, uint32_t b) {
        if (matrix.row_p[a] != matrix.row_p[b]) {
            return matrix.row_p[a] < matrix.row_p[b];
        }
        return matrix.col_p[a] < matrix.col_p[b];
    });
    for (uint32_t i = 0; i < matrix.nnz; i++) {
        this->matrix.row_p[i] = matrix.row_p[perm[i]];
        this->matrix.col_p[i] = matrix.col_p[perm[i]];
        this->matrix.val_p[i] = matrix.val_p[perm[i]];
    }
    return true;
}

template <typename T> typename COO_Cusparse<T>::GPU_Pointers COO_Cusparse<T>::gpu_prep(const T *dense_vec) {
    const uint32_t nnz = this->matrix.nnz;
    GPU_Pointers pointers;
    pointers.matrix = this->matrix;

    cudaMalloc(&pointers.matrix.row_p, nnz * sizeof(uint32_t));
    cudaMalloc(&pointers.matrix.col_p, nnz * sizeof(uint32_t));
    cudaMalloc(&pointers.matrix.val_p, nnz * sizeof(T));
    cudaMalloc(&pointers.dense_vec, this->matrix.cols * sizeof(T));
    cudaMalloc(&pointers.result, this->matrix.rows * sizeof(T));

    cudaMemcpy(pointers.matrix.row_p, this->matrix.row_p, nnz * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(pointers.matrix.col_p, this->matrix.col_p, nnz * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(pointers.matrix.val_p, this->matrix.val_p, nnz * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(pointers.dense_vec, dense_vec, this->matrix.cols * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemset(pointers.result, 0, this->matrix.rows * sizeof(T));

    cusparseCreate(&handle);
    cusparseCreateCoo(&mat_desc, this->matrix.rows, this->matrix.cols, nnz, pointers.matrix.row_p,
                      pointers.matrix.col_p, pointers.matrix.val_p, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                      cusparse_dtype<T>());
    cusparseCreateDnVec(&vec_x, this->matrix.cols, pointers.dense_vec, cusparse_dtype<T>());
    cusparseCreateDnVec(&vec_y, this->matrix.rows, pointers.result, cusparse_dtype<T>());

    // We don't use alpha and beta for now,I think
    const T alpha = static_cast<T>(1);
    const T beta = static_cast<T>(0);

    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_desc, vec_x, &beta, vec_y,
                            cusparse_dtype<T>(), CUSPARSE_SPMV_COO_ALG1, &buffer_size);
    cudaMalloc(&buffer, buffer_size);
    return pointers;
}

template <typename T> std::vector<T> COO_Cusparse<T>::gpu_retrive(const GPU_Pointers &pointers) {
    std::vector<T> result(this->matrix.rows);
    cudaMemcpy(result.data(), pointers.result, this->matrix.rows * sizeof(T), cudaMemcpyDeviceToHost);
    return result;
}

template <typename T> void COO_Cusparse<T>::gpu_free(const GPU_Pointers &pointers) {
    cudaFree(buffer);
    cusparseDestroyDnVec(vec_x);
    cusparseDestroyDnVec(vec_y);
    cusparseDestroySpMat(mat_desc);
    cusparseDestroy(handle);

    cudaFree(pointers.matrix.row_p);
    cudaFree(pointers.matrix.col_p);
    cudaFree(pointers.matrix.val_p);
    cudaFree(pointers.result);
    cudaFree(pointers.dense_vec);
}

template <typename T> COO_Cusparse<T>::~COO_Cusparse() {
    free(this->matrix.val_p);
    free(this->matrix.col_p);
    free(this->matrix.row_p);
}

template class COO_Cusparse<float>;
template class COO_Cusparse<double>;
