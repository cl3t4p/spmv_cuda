#ifndef SPMV_CUDA_UTILS_H
#define SPMV_CUDA_UTILS_H
#include <cmath>
#include <cstdlib>
#include <random>
#include <sys/time.h>
#include <type_traits>
#include <vector>

#define TIMER_DEF struct timeval temp_1, temp_2

#define TIMER_START gettimeofday(&temp_1, (struct timezone *)0)

#define TIMER_STOP gettimeofday(&temp_2, (struct timezone *)0)

#define TIMER_ELAPSED ((temp_2.tv_sec - temp_1.tv_sec) + (temp_2.tv_usec - temp_1.tv_usec) / 1000000.0)

struct Args {
    std::string dtype;
    std::string matrix_type;
    std::string path;
    bool measure_conversion = false;
    int gpu_warmup = 10;
    int gpu_runs = 100;
    int conv_warmup = 2;
    int conv_runs = 5;
    int seed = -1; // -1 = pick a random seed via std::random_device
};

template <typename T> double diff_vector(const std::vector<T> &a, const std::vector<T> &b) {
    if (a.size() != b.size()) return -1;
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        den += std::abs(static_cast<double>(a[i]));
    }
    return den > 0.0 ? num / den : num;
}


template <typename Matrix, typename T> double time_conversion(const COO_Matrix<T> &coo, int warmup, int runs) {
    TIMER_DEF;
    for (int i = 0; i < warmup; ++i) {
        Matrix tmp;
        tmp.load_from_coo(coo);
    }
    double total = 0.0;
    for (int i = 0; i < runs; ++i) {
        Matrix tmp;
        TIMER_START;
        tmp.load_from_coo(coo);
        TIMER_STOP;
        total += TIMER_ELAPSED;
    }
    return (runs > 0) ? total / static_cast<double>(runs) : 0.0;
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

template <typename T> std::vector<T> generate_dense_vec(uint len, int seed) {
    std::vector<T> v(len);
    std::mt19937 gen(static_cast<std::uint32_t>(seed));
    if constexpr (std::is_same_v<T, int>) {
        std::uniform_int_distribution dist(-10, 10);
        for (uint i = 0; i < len; ++i)
            v[i] = dist(gen);
    } else {
        std::uniform_real_distribution<T> dist(static_cast<T>(-1), static_cast<T>(1));
        for (uint i = 0; i < len; ++i)
            v[i] = dist(gen);
    }
    return v;
}

template <typename T>
void printInfo(cudaDeviceProp props, const Args &args, const COO_Matrix<T> &mat, const LaunchConfig &l_conf) {
    printf("================================== Device Info "
           "==================================\n");
    uint nnz = mat.nnz;
    uint cols = mat.cols;
    uint rows = mat.rows;

    //Peak theoretical memory bandwidth in GB/s, from device properties.
    int mem_clock_khz = 0;
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0);
    const double clock_hz = static_cast<double>(mem_clock_khz) * 1e3;
    const double bus_bytes = static_cast<double>(props.memoryBusWidth) / 8.0;
    const double peak_bw = (clock_hz * bus_bytes * 2.0) / 1e9;

    const double avg_nnz_per_row = static_cast<double>(nnz) / rows;
    printf("Device      : %s, peak BW = %.1f GB/s\n", props.name, peak_bw);
    printf("dtype       : %s\n", args.dtype.c_str());
    printf("kernel      : %s\n", args.matrix_type.c_str());
    printf("mtx_input   : %s\n", args.path.c_str());
    printf("Matrix      : rows = %u, cols = %u, nnz = %u, avg nnz/row = %.2f\n", rows, cols, nnz, avg_nnz_per_row);
    printf("Launch      : blk_size = %u, grd_size = %u, ", l_conf.block_size, l_conf.grid_size);
    if (l_conf.shared_bytes > 0) {
        printf("shared : %lu bytes\n", l_conf.shared_bytes);
    }
}
#endif // SPMV_CUDA_UTILS_H
