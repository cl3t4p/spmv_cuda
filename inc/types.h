#ifndef SPMV_CUDA_TYPES_H
#define SPMV_CUDA_TYPES_H
#include <cstdint>
#include <string>
#include <vector>

template <typename T, typename MatrixFormat> struct T_GPU_Pointers {
    T *result;
    T *dense_vec;
    MatrixFormat matrix;
};

struct BASE_Matrix {
    uint32_t rows;
    uint32_t cols;
    uint32_t nnz;
};

template <typename T> struct COO_Matrix : BASE_Matrix {
    uint32_t *row_p;
    uint32_t *col_p;
    T *val_p;
};

template <typename T> struct CSR_Matrix : BASE_Matrix {
    uint32_t *row_ptr;
    uint32_t *col_idx;
    T *val_p;
};

template <typename T, template <typename> class MatrixFormat> class SparseMatrixGPU {
    static_assert(std::is_same_v<T, int> || std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "T must be int, float, or double");

  public:
    using MatrixType = MatrixFormat<T>;
    using GPU_Pointers = T_GPU_Pointers<T, MatrixType>;

  protected:
    MatrixType matrix = {};

  public:
    virtual ~SparseMatrixGPU() = default;

    virtual bool load_from_file(const std::string &path) = 0;

    // GPU Stuff
    virtual GPU_Pointers gpu_prep(const T *dense_vec) const;

    virtual void gpu_compute(GPU_Pointers *pointers, uint grid_size, uint blk_size);

    virtual std::vector<T> gpu_retrive(const GPU_Pointers &pointers);

    virtual void gpu_free(const GPU_Pointers &pointers);

    void gpu_free_result(GPU_Pointers *pointers) { cudaMemset(pointers->result, 0, getRows() * sizeof(T)); }

    virtual COO_Matrix<T> get_coo_matrix();

    [[nodiscard]] uint32_t getRows() const { return matrix.rows; }
    [[nodiscard]] uint32_t getCols() const { return matrix.cols; }
    [[nodiscard]] uint32_t getNnz() const { return matrix.nnz; }
    [[nodiscard]] MatrixType getMatrix() const { return matrix; }
};

#endif // SPMV_CUDA_TYPES_H
