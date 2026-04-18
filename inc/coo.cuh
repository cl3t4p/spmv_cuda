#ifndef SPMV_MATRIX_MARKET_H
#define SPMV_MATRIX_MARKET_H
#include <cstdint>
#include <string>
#include <sys/types.h>
#include <type_traits>
#include <vector>

template <typename T> class COO {
public:
  typedef struct {
    u_int32_t rows;
    u_int32_t cols;
    u_int32_t nnz;
    u_int32_t *row_p;
    u_int32_t *col_p;
    T *val_p;
  } COO_Matrix;

  typedef struct {
    T *result;
    T *dense_vec;
    COO_Matrix matrix;
  } GPU_COO_Pointers;

  static_assert(std::is_same_v<T, int> || std::is_same_v<T, float> ||
                    std::is_same_v<T, double>,
                "T must be int, float, or double");

private:
  COO_Matrix matrix = {};

public:
  ~COO();

  GPU_COO_Pointers gpu_prep(const T *dense_vec) const;
  [[nodiscard]] std::vector<T>
  gpu_retrive(const GPU_COO_Pointers &pointers) const;

  void gpu_compute(GPU_COO_Pointers *pointers, uint grid_size, uint blk_size);

  void free_result(GPU_COO_Pointers *pointers) {
    cudaMemset(pointers->result, 0, getRows() * sizeof(T));
  }

  static void gpu_free(const GPU_COO_Pointers &pointers);
  bool load_from_file(const std::string &path);
  [[nodiscard]] std::vector<T>
  cpu_compute(const std::vector<T> &dense_vec) const;

  [[nodiscard]] uint32_t getRows() const { return matrix.rows; }
  [[nodiscard]] uint32_t getCols() const { return matrix.cols; }
  [[nodiscard]] uint32_t getNnz() const { return matrix.nnz; }
  [[nodiscard]] COO_Matrix getMatrix() const { return matrix; }
};

template <typename T>
__global__ void spmv_coo_kernel(typename COO<T>::COO_Matrix matrix,
                                const T *dense_vec, T *result);

#endif // SPMV_MATRIX_MARKET_H
