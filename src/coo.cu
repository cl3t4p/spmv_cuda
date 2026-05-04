#include "coo.cuh"
#include "spm_loader.cuh"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <tuple>
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

// Warp-segmented reduction in shared memory. Requires the COO sorted by row.
// Each thread loads one nnz product into shared memory, then a Hillis-Steele scan
// masked by row equality sums contiguous same-row entries inside the warp using
// only shared-memory adds (no intra-warp atomics). The thread at the end of each
// row segment is the sole writer; atomicAdd is used only at row-boundaries that
// cross warps (one atomic per segment, not per nnz).
template <typename T> __global__ void spmv_coo_optimized_kernel(COO_Matrix<T> matrix, const T *dense_vec, T *result) {
    extern __shared__ unsigned char smem_raw[];
    T *s_val = reinterpret_cast<T *>(smem_raw);
    uint32_t *s_row = reinterpret_cast<uint32_t *>(s_val + blockDim.x);

    const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint lane = threadIdx.x & 31u;
    const bool active = tid < matrix.nnz;

    const uint32_t row = active ? matrix.row_p[tid] : 0xFFFFFFFFu;
    T val = T(0);
    if (active) {
        val = matrix.val_p[tid] * dense_vec[matrix.col_p[tid]];
    }

    s_row[threadIdx.x] = row;
    s_val[threadIdx.x] = val;
    __syncwarp();

    // Hillis-Steele segmented scan within the warp using shared memory.
    for (int off = 1; off < 32; off <<= 1) {
        T add = T(0);
        if (lane >= static_cast<uint>(off) && s_row[threadIdx.x - off] == row) {
            add = s_val[threadIdx.x - off];
        }
        __syncwarp();
        s_val[threadIdx.x] += add;
        __syncwarp();
    }

    // Segment end = last lane of warp, or next lane has a different row, or last nnz.
    const bool last_lane = (lane == 31u);
    const uint32_t next_row = last_lane ? 0xFFFFFFFEu : s_row[threadIdx.x + 1];
    const bool segment_end = active && (last_lane || next_row != row || tid + 1 == matrix.nnz);

    if (segment_end) {
        // Same row may continue in the next warp/block, so atomic is required here.
        // But this fires once per segment, not once per nnz.
        atomicAdd(&result[row], s_val[threadIdx.x]);
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

template <typename T> bool COO_Optimized<T>::load_from_coo(const COO_Matrix<T> &matrix) {
    if (!COO<T>::load_from_coo(matrix)) {
        return false;
    }
    // Segmented-reduction kernel needs entries sorted by row so same-row nnz are
    // contiguous within each warp.
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

// Explicit instantiations. atomicAdd supports int, float, double (sm_60+).
template class COO<int>;
template class COO<float>;

template class COO_Optimized<int>;
template class COO_Optimized<float>;

template __global__ void spmv_coo_kernel<int>(COO_Matrix<int>, const int *, int *);
template __global__ void spmv_coo_kernel<float>(COO_Matrix<float>, const float *, float *);

template __global__ void spmv_coo_optimized_kernel<int>(COO_Matrix<int>, const int *, int *);
template __global__ void spmv_coo_optimized_kernel<float>(COO_Matrix<float>, const float *, float *);

