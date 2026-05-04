#include <cuda_runtime_api.h>
#include <iostream>
#include <ostream>
#include <string>
#include <sys/time.h>
#include <type_traits>
#include <vector>

#include "argparse.h"
#include "coo.cuh"
#include "csr.cuh"
#include "cusparse_spmv.cuh"
#include "ell.cuh"
#include "spm_loader.cuh"
#include "stats.h"
#include "utils.h"

struct Args {
    std::string dtype;
    std::string matrix_type;
    std::string path;
    bool measure_conversion = false;
    int gpu_warmup = 10;
    int gpu_runs = 100;
    int conv_warmup = 2;
    int conv_runs = 5;
};

template <typename Matrix, typename T> int run(const Args &args) {
    COO_Matrix<T> coo_matrix{};

    std::cout << "reading mtx file" << std::endl;
    if (!MatrixMarketLoader<T>::load(args.path, coo_matrix)) {
        std::cerr << "Error loading matrix from " << args.path << std::endl;
        return -1;
    }

    double conversion_time = 0.0;
    if (args.measure_conversion) {
        std::cout << "timing COO -> " << args.matrix_type << " conversion (" << args.conv_warmup << " warmup, "
                  << args.conv_runs << " runs)" << std::endl;
        conversion_time = time_conversion<Matrix, T>(coo_matrix, args.conv_warmup, args.conv_runs);
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

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, 0);

    LaunchConfig launch_config = mat.getLaunchConfig();

    uint blk_size = launch_config.block_size;
    uint grd_size = launch_config.grid_size;

    auto gpu_pointers = mat.gpu_prep(dense_vec.data());
    std::cout << "GPU : Compute" << std::endl;

    for (int x = 0; x < args.gpu_warmup; x++) {
        mat.gpu_free_result(&gpu_pointers);
        mat.gpu_compute(&gpu_pointers, grd_size, blk_size);
        cudaDeviceSynchronize();
    }

    for (int x = 0; x < args.gpu_runs; x++) {
        mat.gpu_free_result(&gpu_pointers);

        cudaEventRecord(start_event);
        mat.gpu_compute(&gpu_pointers, grd_size, blk_size);
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);

        float kernel_ms = 0.0f;
        cudaEventElapsedTime(&kernel_ms, start_event, stop_event);
        gpu_times.push_back(kernel_ms / 1000.0f);
    }
    auto gpu_result = mat.gpu_retrive(gpu_pointers);
    mat.gpu_free(gpu_pointers);

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
#if DEBUG
    std::cout << "Comparing" << std::endl;

    bool result = compare_vectors<T>(cpu_result, gpu_result);


    if (result) {
//    std::cout << "2 Vector are similar" << std::endl;
    } else {
  //      std::cout << "2 Vector not are similar!!!" << std::endl;
    }
#endif
    error = diff_vector<T>(cpu_result, gpu_result);

    const double gpu_amean = arithmetic_mean(gpu_times);
    const double gpu_gmean = geometric_mean(gpu_times);
    const double gpu_sigma = stddev(gpu_times, gpu_amean);
    gputime = static_cast<float>(gpu_amean);

    const double flops = 2.0 * static_cast<double>(mat.getMatrix().nnz);
    const double cpu_gflops = (cputime > 0.0) ? (flops / cputime) / 1e9 : 0.0;
    const double gpu_gflops = (gputime > 0.0) ? (flops / gputime) / 1e9 : 0.0;

    printf("================================== Times and results of my code "
           "==================================\n");
    printf("dtype = %s, kernel = %s\n", args.dtype.c_str(), args.matrix_type.c_str());
    printf("Error between CPU and GPU is %.15e\n", error);
    printf("\nVector len = %d, CPU time = %5.3e sec, CPU GFlops = %5.3f\n", mat.getMatrix().nnz, cputime, cpu_gflops);
    printf("\nblk_size = %d, grd_size = %d, GPU time (cudaEvent): %5.3e sec, GPU GFlops = %5.3f\n", blk_size, grd_size,
           gputime, gpu_gflops);
    printf("GPU stats over %d runs: arith_mean = %5.3e sec, geo_mean = %5.3e sec, stddev = %5.3e sec\n", args.gpu_runs,
           gpu_amean, gpu_gmean, gpu_sigma);
    if (args.measure_conversion) {
        printf("\nCOO -> %s conversion time = %5.3e sec (mean of %d runs, %d warmup)\n", args.matrix_type.c_str(),
               conversion_time, args.conv_runs, args.conv_warmup);
    }
    return 0;
}

template <typename T> int run_cusparse(const Args &args) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        if (args.matrix_type == "csr_cusparse") {
            return run<CSR_CuSparse<T>, T>(args);
        }
        if (args.matrix_type == "coo_cusparse") {
            return run<COO_CuSparse<T>, T>(args);
        }
    } else {
        std::cerr << "cuSPARSE backend supports only float/double" << std::endl;
        return 1;
    }
    return -1;
}

template <typename T> int run_by_format(const Args &args) {
    const std::string &mt = args.matrix_type;
    if (mt == "coo")
        return run<COO<T>, T>(args);
    if (mt == "coo_opt")
        return run<COO_Optimized<T>, T>(args);
    if (mt == "csr_scalar")
        return run<CSR_Scalar<T>, T>(args);
    if (mt == "csr_vec")
        return run<CSR_Vector<T>, T>(args);
    if (mt == "ell")
        return run<ELL<T>, T>(args);
    if (mt == "csr_cusparse" || mt == "coo_cusparse")
        return run_cusparse<T>(args);
    std::cerr << "Unknown matrix type: " << mt << std::endl;
    return 1;
}

int main(int argc, char **argv) {
    Args args;
    argparse::Parser p;
    p.description = "SpMV CUDA benchmark.";
    p.add_positional("dtype", "value type", args.dtype, {"int", "float", "double"});
    p.add_positional("matrix_type", "storage format / kernel", args.matrix_type,
                     {"coo", "coo_opt", "csr_scalar", "csr_vec", "ell", "csr_cusparse", "coo_cusparse"});
    p.add_positional("file", "path to .mtx file", args.path);
    p.add_flag("--conversion", "time the COO -> matrix_type host conversion and report it at the end",
               args.measure_conversion);
    p.add_int("--gpu-warmup", "GPU warmup launches (default 10)", args.gpu_warmup);
    p.add_int("--gpu-runs", "GPU timed launches (default 100)", args.gpu_runs);
    p.add_int("--conv-warmup", "conversion warmup runs (default 2)", args.conv_warmup);
    p.add_int("--conv-runs", "conversion timed runs (default 5)", args.conv_runs);
    p.parse(argc, argv);

    if (args.dtype == "int")
        return run_by_format<int>(args);
    if (args.dtype == "float")
        return run_by_format<float>(args);
    return 1;
}
