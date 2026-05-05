#include "cusparse_spmv.cuh"
#include <algorithm>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <vector>

#define CUSPARSE_CHECK(call)                                                                                           \
    do {                                                                                                               \
        cusparseStatus_t s_ = (call);                                                                                  \
        if (s_ != CUSPARSE_STATUS_SUCCESS) {                                                                           \
            std::cerr << "cuSPARSE error " << s_ << " at " << __FILE__ << ":" << __LINE__ << " : "                     \
                      << cusparseGetErrorString(s_) << std::endl;                                                      \
        }                                                                                                              \
    } while (0)

// For type conversion
template <typename T> struct AlphaBeta {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "cuSPARSE SpMV here is restricted to float/double");
    T alpha = static_cast<T>(1);
    T beta = static_cast<T>(0);
};

template <typename T> typename CSR_CuSparse<T>::GPU_Pointers CSR_CuSparse<T>::gpu_prep(const T *dense_vec) {
    GPU_Pointers pointers = CSR<T>::gpu_prep(dense_vec);

    CUSPARSE_CHECK(cusparseCreate(&handle));

    CUSPARSE_CHECK(cusparseCreateCsr(&mat_desc, this->matrix.rows, this->matrix.cols, this->matrix.nnz,
                                     pointers.matrix.row_ptr, pointers.matrix.col_idx, pointers.matrix.val_p,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                     cusparse_value_type<T>()));

    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_x, this->matrix.cols, pointers.dense_vec, cusparse_value_type<T>()));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_y, this->matrix.rows, pointers.result, cusparse_value_type<T>()));

    AlphaBeta<T> ab;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &ab.alpha, mat_desc, vec_x,
                                           &ab.beta, vec_y, cusparse_value_type<T>(), CUSPARSE_SPMV_CSR_ALG1,
                                           &buffer_size));
    if (buffer_size > 0) {
        cudaMalloc(&d_buffer, buffer_size);
    }
    return pointers;
}

template <typename T> void CSR_CuSparse<T>::gpu_compute(GPU_Pointers *pointers, uint, uint) {
    AlphaBeta<T> ab;
    CUSPARSE_CHECK(cusparseDnVecSetValues(vec_y, pointers->result));
    CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &ab.alpha, mat_desc, vec_x, &ab.beta, vec_y,
                                cusparse_value_type<T>(), CUSPARSE_SPMV_CSR_ALG1, d_buffer));
}

template <typename T> void CSR_CuSparse<T>::gpu_free(const GPU_Pointers &pointers) {
    if (vec_y) {
        cusparseDestroyDnVec(vec_y);
        vec_y = nullptr;
    }
    if (vec_x) {
        cusparseDestroyDnVec(vec_x);
        vec_x = nullptr;
    }
    if (mat_desc) {
        cusparseDestroySpMat(mat_desc);
        mat_desc = nullptr;
    }
    if (handle) {
        cusparseDestroy(handle);
        handle = nullptr;
    }
    if (d_buffer) {
        cudaFree(d_buffer);
        d_buffer = nullptr;
        buffer_size = 0;
    }
    CSR<T>::gpu_free(pointers);
}

template <typename T> bool COO_CuSparse<T>::load_from_coo(const COO_Matrix<T> &matrix) {
    if (!COO<T>::load_from_coo(matrix)) {
        return false;
    }
    // cuSPARSE COO SpMV (ALG1) requires entries sorted by row index.
    const uint32_t nnz = this->matrix.nnz;
    std::vector<uint32_t> perm(nnz);
    std::iota(perm.begin(), perm.end(), 0u);
    auto *row_p = this->matrix.row_p;
    auto *col_p = this->matrix.col_p;
    auto *val_p = this->matrix.val_p;
    std::sort(perm.begin(), perm.end(), [row_p, col_p](uint32_t a, uint32_t b) {
        return std::tie(row_p[a], col_p[a]) < std::tie(row_p[b], col_p[b]);
    });

    std::vector<uint32_t> r(nnz), c(nnz);
    std::vector<T> v(nnz);
    for (uint32_t i = 0; i < nnz; ++i) {
        r[i] = row_p[perm[i]];
        c[i] = col_p[perm[i]];
        v[i] = val_p[perm[i]];
    }
    std::copy(r.begin(), r.end(), row_p);
    std::copy(c.begin(), c.end(), col_p);
    std::copy(v.begin(), v.end(), val_p);
    return true;
}

template <typename T> typename COO_CuSparse<T>::GPU_Pointers COO_CuSparse<T>::gpu_prep(const T *dense_vec) {
    GPU_Pointers pointers = COO<T>::gpu_prep(dense_vec);

    CUSPARSE_CHECK(cusparseCreate(&handle));

    CUSPARSE_CHECK(cusparseCreateCoo(&mat_desc, this->matrix.rows, this->matrix.cols, this->matrix.nnz,
                                     pointers.matrix.row_p, pointers.matrix.col_p, pointers.matrix.val_p,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, cusparse_value_type<T>()));

    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_x, this->matrix.cols, pointers.dense_vec, cusparse_value_type<T>()));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_y, this->matrix.rows, pointers.result, cusparse_value_type<T>()));

    AlphaBeta<T> ab;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &ab.alpha, mat_desc, vec_x,
                                           &ab.beta, vec_y, cusparse_value_type<T>(), CUSPARSE_SPMV_COO_ALG1,
                                           &buffer_size));
    if (buffer_size > 0) {
        cudaMalloc(&d_buffer, buffer_size);
    }
    return pointers;
}

template <typename T> void COO_CuSparse<T>::gpu_compute(GPU_Pointers *pointers, uint, uint) {
    AlphaBeta<T> ab;
    CUSPARSE_CHECK(cusparseDnVecSetValues(vec_y, pointers->result));
    CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &ab.alpha, mat_desc, vec_x, &ab.beta, vec_y,
                                cusparse_value_type<T>(), CUSPARSE_SPMV_COO_ALG1, d_buffer));
}

template <typename T> void COO_CuSparse<T>::gpu_free(const GPU_Pointers &pointers) {
    if (vec_y) {
        cusparseDestroyDnVec(vec_y);
        vec_y = nullptr;
    }
    if (vec_x) {
        cusparseDestroyDnVec(vec_x);
        vec_x = nullptr;
    }
    if (mat_desc) {
        cusparseDestroySpMat(mat_desc);
        mat_desc = nullptr;
    }
    if (handle) {
        cusparseDestroy(handle);
        handle = nullptr;
    }
    if (d_buffer) {
        cudaFree(d_buffer);
        d_buffer = nullptr;
        buffer_size = 0;
    }
    COO<T>::gpu_free(pointers);
}

template class CSR_CuSparse<float>;
template class COO_CuSparse<float>;
