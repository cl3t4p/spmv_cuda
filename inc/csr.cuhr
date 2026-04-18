#ifndef SPMV_CSR_CUH
#define SPMV_CSR_CUH
#include <cstdint>
#include <string>
#include <sys/types.h>
#include <type_traits>
#include <vector>

template <typename T> class CSR {
public:
  typedef struct {
    uint32_t rows;
    uint32_t cols;
    uint32_t nnz;
    uint32_t *row_ptr; // size rows + 1
    uint32_t *col_idx; // size nnz
    T *val_p;          // size nnz
  } CSR_Matrix;

  typedef struct {
    T *result;
    T *dense_vec;
    CSR_Matrix matrix;
  } GPU_CSR_Pointers;

  static_assert(std::is_same_v<T, int> || std::is_same_v<T, float> ||
                    std::is_same_v<T, double>,
                "T must be int, float, or double");

private:
  CSR_Matrix matrix = {};

public:
  ~CSR();

  GPU_CSR_Pointers gpu_prep(const T *dense_vec) const;
  [[nodiscard]] std::vector<T>
  gpu_retrive(const GPU_CSR_Pointers &pointers) const;

  void gpu_compute(GPU_CSR_Pointers *pointers, uint grid_size, uint blk_size);

  void free_result(GPU_CSR_Pointers *pointers) {
    cudaMemset(pointers->result, 0, getRows() * sizeof(T));
  }

  static void gpu_free(const GPU_CSR_Pointers &pointers);
  bool load_from_file(const std::string &path);
  [[nodiscard]] std::vector<T>
  cpu_compute(const std::vector<T> &dense_vec) const;

  [[nodiscard]] uint32_t getRows() const { return matrix.rows; }
  [[nodiscard]] uint32_t getCols() const { return matrix.cols; }
  [[nodiscard]] uint32_t getNnz() const { return matrix.nnz; }
  [[nodiscard]] CSR_Matrix getMatrix() const { return matrix; }
};

template <typename T>
__global__ void spmv_csr_kernel(typename CSR<T>::CSR_Matrix matrix,
                                const T *dense_vec, T *result);

#endif // SPMV_CSR_CUH
