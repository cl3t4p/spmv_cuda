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

struct LaunchConfig {
    uint grid_size;
    uint block_size;
    size_t shared_bytes; // 0 if the kernel uses no dynamic shared memory
};

template <typename T, template <typename> class MatrixFormat> class SparseMatrixGPU {
    static_assert(std::is_same_v<T, int> || std::is_same_v<T, float> ,
                  "T must be int, float");

  public:
    using MatrixType = MatrixFormat<T>;
    using GPU_Pointers = T_GPU_Pointers<T, MatrixType>;

  protected:
    MatrixType matrix = {};
    LaunchConfig launch_config = {};
    virtual void calculate_launch_config() = 0;

  public:
    virtual ~SparseMatrixGPU() = default;

    virtual bool load_from_coo(const COO_Matrix<T> &matrix) = 0;

    // GPU Stuff
    virtual GPU_Pointers gpu_prep(const T *dense_vec) = 0;

    virtual void gpu_compute(GPU_Pointers *, uint, uint) = 0;

    virtual void gpu_free(const GPU_Pointers &pointers) = 0;

    std::vector<T> gpu_retrive(const GPU_Pointers &pointers) {
        std::vector<T> result(this->matrix.rows);
        cudaMemcpy(result.data(), pointers.result, this->matrix.rows * sizeof(T), cudaMemcpyDeviceToHost);
        return result;
    }

    void gpu_free_result(GPU_Pointers *pointers) { cudaMemset(pointers->result, 0, getRows() * sizeof(T)); }

    [[nodiscard]] LaunchConfig getLaunchConfig() {
        if (launch_config.grid_size == 0) {
            calculate_launch_config();
        }
        return launch_config;
    }
    [[nodiscard]] uint32_t getRows() const { return matrix.rows; }
    [[nodiscard]] uint32_t getCols() const { return matrix.cols; }
    [[nodiscard]] uint32_t getNnz() const { return matrix.nnz; }
    [[nodiscard]] MatrixType getMatrix() const { return matrix; }
};

#endif // SPMV_CUDA_TYPES_H
