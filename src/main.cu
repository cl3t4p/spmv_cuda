#include <cuda_runtime_api.h>
#include <iostream>
#include <ostream>
#include <string>
#include <sys/time.h>
#include <vector>

#include "coo.cuh"
#include "csr.cuh"
#include "ell.cuh"
#include "spm_loader.cuh"
#include "utils.h"

template <typename Matrix, typename T> int run(const char *path) {
    COO_Matrix<T> coo_matrix{};

    std::cout << "reading mtx file" << std::endl;
    if (!MatrixMarketLoader<T>::load(path, coo_matrix)) {
        std::cerr << "Error loading matrix from " << path << std::endl;
        return -1;
    }
    Matrix mat;

    std::cout << "loading coo matrix" << std::endl;
    if (!mat.load_from_coo(coo_matrix)) {
        std::cerr << "Error converting coo matrix" << std::endl;
        return -1;
    }
    std::cout << "matrix loaded with success!" << std::endl;

    std::vector<T> dense_vec(mat.getCols(), static_cast<T>(1));

    TIMER_DEF;
    std::vector<float> gpu_times;
    float cputime, gputime;
    double error;

    std::cout << "CPU : Compute" << std::endl;
    TIMER_START;
    auto cpu_result = cpu_compute<T>(coo_matrix, dense_vec);
    TIMER_STOP;

    cputime = TIMER_ELAPSED;

    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, 0);

    LaunchConfig launch_config = mat.getLaunchConfig();

    uint blk_size = launch_config.block_size;
    uint grd_size = launch_config.grid_size;
    // TODO For memory shared devices
    uint shared_size = launch_config.shared_bytes;

    auto gpu_pointers = mat.gpu_prep(dense_vec.data());
    std::cout << "GPU : Compute" << std::endl;

    // warmup
    for (int x = 0; x < 10; x++) {
        mat.gpu_free_result(&gpu_pointers);
        mat.gpu_compute(&gpu_pointers, grd_size, blk_size);
        cudaDeviceSynchronize();
    }

    // benchmark
    for (int x = 0; x < 100; x++) {
        mat.gpu_free_result(&gpu_pointers);

        TIMER_START;
        mat.gpu_compute(&gpu_pointers, grd_size, blk_size);
        cudaDeviceSynchronize();
        TIMER_STOP;
        gpu_times.push_back(TIMER_ELAPSED);
    }
    auto gpu_result = mat.gpu_retrive(gpu_pointers);
    mat.gpu_free(gpu_pointers);

    std::cout << "Comparing" << std::endl;

    bool result = compare_vectors<T>(cpu_result, gpu_result);
    error = diff_vector<T>(cpu_result, gpu_result);

    if (result) {
        std::cout << "2 Vector are similar" << std::endl;
    } else {
        std::cout << "2 Vector not are similar!!!" << std::endl;
    }

    gputime = 0;
    for (auto time : gpu_times) {
        gputime += time;
    }
    gputime = gputime / gpu_times.size();

    printf("================================== Times and results of my code "
           "==================================\n");
    printf("Error between CPU and GPU is %.15e\n", error);
    printf("\nVector len = %d, CPU time = %5.3f\n", mat.getMatrix().nnz, cputime);
    printf("\nblk_size = %d, grd_size = %d, GPU time (gettimeofday): %5.3f sec\n", blk_size, grd_size, gputime);
    return 0;

    MatrixMarketLoader<T>::free_matrix(coo_matrix);
}

const std::string s_matrix_type = "(coo | csr_scalar | csr_vec | ell)";
const std::string s_dtype = "(int | float | double)";

template <typename T> int run_by_format(const std::string &matrix_type, const char *path) {
    if (matrix_type == "coo") {
        return run<COO<T>, T>(path);
    }
    if (matrix_type == "csr_scalar") {
        return run<CSR_Scalar<T>, T>(path);
    }
    if (matrix_type == "csr_vec") {
        return run<CSR_Vector<T>, T>(path);
    }
    if (matrix_type == "ell") {
        return run <ELL<T>, T>(path);
    }
    std::cerr << "Unknown matrix type: " << matrix_type << s_matrix_type << std::endl;
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <dtype> <matrix_type> <file>" << std::endl;
        std::cout << "  dtype: " << s_dtype << std::endl;
        std::cout << "  matrix_type:" << s_matrix_type << std::endl;
        return 1;
    }

    const std::string dtype = argv[1];
    const std::string matrix_type = argv[2];
    const char *path = argv[3];

    if (dtype == "int") {
        return run_by_format<int>(matrix_type, path);
    }
    if (dtype == "float") {
        return run_by_format<float>(matrix_type, path);
    }
    if (dtype == "double") {
        return run_by_format<double>(matrix_type, path);
    }
    std::cerr << "Unknown dtype: " << dtype << " (expected int, float, or double)" << std::endl;
    return 1;
}
