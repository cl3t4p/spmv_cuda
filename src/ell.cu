#include "csr.cuh"
#include "ell.cuh"

template <typename T> __global__ void spmv_ell_kernel(ELL_Matrix<T> matrix, const T *dense_vec, T *result) {
    uint row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= matrix.rows)
        return;

    T sum = 0;

    for (int k = 0; k < matrix.max_col_len; k++) {
        int idx = k * matrix.rows + row; // column major
        int col = matrix.col_idx[idx];
        T val = matrix.values[idx];
        sum += val * dense_vec[col];
    }

    result[row] = sum;
}

template <typename T> bool ELL<T>::load_from_coo(const COO_Matrix<T> &og_matrix) {
    CSR_Scalar<T> csr;
    if (!csr.load_from_coo(og_matrix)) {
        return false;
    }
    auto csr_mat = csr.getMatrix();

    ELL_Matrix<T> mat{};
    mat.rows = csr_mat.rows;
    mat.cols = csr_mat.cols;
    mat.nnz = csr_mat.nnz;

    const uint n_rows = mat.rows;
    uint max_col_len = 0;

    // Find max rows len for ELL matrix
    for (uint i = 0; i < n_rows; i++) {
        uint len = csr_mat.row_ptr[i + 1] - csr_mat.row_ptr[i];
        max_col_len = std::max(max_col_len, len);
    }

    mat.col_idx = static_cast<uint *>(calloc(n_rows * (max_col_len), sizeof(uint)));
    mat.values = static_cast<T *>(calloc(n_rows * (max_col_len), sizeof(T)));
    mat.max_col_len = max_col_len;

    for (uint row = 0; row < n_rows; row++) {
        for (uint col = csr_mat.row_ptr[row]; col < csr_mat.row_ptr[row + 1]; col++) {
            const uint index = col - csr_mat.row_ptr[row];
            // Column major
            mat.col_idx[index * n_rows + row] = csr_mat.col_idx[col];
            mat.values[index * n_rows + row] = csr_mat.val_p[col];
        }
    }

    this->matrix = mat;
    return true;
}

template <typename T> ELL<T>::~ELL() {
    free(this->matrix.values);
    free(this->matrix.col_idx);
}

template <typename T> typename ELL<T>::GPU_Pointers ELL<T>::gpu_prep(const T *dense_vec) {
    GPU_Pointers pointers;
    pointers.matrix = this->matrix;

    cudaMalloc(&pointers.dense_vec, this->matrix.cols * sizeof(T));
    cudaMalloc(&pointers.result, this->matrix.rows * sizeof(T));

    const uint max_col_len = this->matrix.max_col_len;
    const uint n_rows = this->matrix.rows;
    const uint size_matrix = max_col_len * n_rows;

    cudaMalloc(&pointers.matrix.col_idx, size_matrix * sizeof(uint));
    cudaMalloc(&pointers.matrix.values, size_matrix * sizeof(T));

    cudaMemcpy(pointers.matrix.col_idx, this->matrix.col_idx, size_matrix * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(pointers.matrix.values, this->matrix.values, size_matrix * sizeof(T), cudaMemcpyHostToDevice);

    cudaMemcpy(pointers.dense_vec, dense_vec, this->matrix.cols * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemset(pointers.result, 0, this->matrix.rows * sizeof(T));
    return pointers;
}

template <typename T> void ELL<T>::gpu_free(const GPU_Pointers &pointers) {
    cudaFree(pointers.matrix.col_idx);
    cudaFree(pointers.matrix.values);
    cudaFree(pointers.result);
    cudaFree(pointers.dense_vec);
}

template class ELL<int>;
template class ELL<float>;

template __global__ void spmv_ell_kernel<int>(ELL_Matrix<int>, const int *, int *);
template __global__ void spmv_ell_kernel<float>(ELL_Matrix<float>, const float *, float *);