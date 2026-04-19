#ifndef SPMV_CUDA_UTILS_H
#define SPMV_CUDA_UTILS_H
#include <cmath>
#include <cstdlib>
#include <type_traits>
#include <vector>

#define TIMER_DEF struct timeval temp_1, temp_2

#define TIMER_START gettimeofday(&temp_1, (struct timezone *)0)

#define TIMER_STOP gettimeofday(&temp_2, (struct timezone *)0)

#define TIMER_ELAPSED ((temp_2.tv_sec - temp_1.tv_sec) + (temp_2.tv_usec - temp_1.tv_usec) / 1000000.0)

template <typename T> constexpr double default_eps() {
    if constexpr (std::is_same_v<T, int>) {
        return 0.0;
    } else if constexpr (std::is_same_v<T, float>) {
        return 1e-4;
    } else {
        return 1e-9;
    }
}

template <typename T>
bool compare_vectors(const std::vector<T> &a, const std::vector<T> &b, const double eps = default_eps<T>()) {
    static_assert(std::is_same_v<T, int> || std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "compare_vectors supports int, float, double");
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        const double diff = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (diff > eps) {
            return false;
        }
    }
    return true;
}

template <typename T> double diff_vector(const std::vector<T> &a, const std::vector<T> &b) {
    if (a.size() != b.size()) {
        return -1;
    }
    double result = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += std::abs(a[i] - b[i]);
    }
    return result / static_cast<double>(a.size());
}

template <typename T> std::vector<T> cpu_compute(const COO_Matrix<T> &coo_matrix, const std::vector<T> &dense_vec) {
    std::vector<T> result(coo_matrix.rows, 0);
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < coo_matrix.nnz; i++) {
#pragma omp atomic
        result[coo_matrix.row_p[i]] += coo_matrix.val_p[i] * dense_vec[coo_matrix.col_p[i]];
    }
    return result;
}

#endif // SPMV_CUDA_UTILS_H
