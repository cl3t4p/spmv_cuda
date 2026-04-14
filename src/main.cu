#include <iostream>
#include <ostream>
#include <vector>
#include <algorithm>
#include <time.h>
#include <sys/time.h>

#include "matrix_market.h"

#define TIMER_DEF     struct timeval temp_1, temp_2
#define TIMER_START   gettimeofday(&temp_1, (struct timezone*)0)
#define TIMER_STOP    gettimeofday(&temp_2, (struct timezone*)0)
#define TIMER_ELAPSED ((temp_2.tv_sec-temp_1.tv_sec)+(temp_2.tv_usec-temp_1.tv_usec)/1000000.0)



bool compare_vectors(const std::vector<double>& a,
                     const std::vector<double>& b,
                     double eps = 1e-6) {
    if (a.size() != b.size()) {
        return false;
    }

    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > eps) {
            return false;
        }
    }
    return true;
}


double diff_vector(const std::vector<double>& a,
                     const std::vector<double>& b) {
    if (a.size() != b.size()) {
        return -1;
    }
    double result = 0;
    for (size_t i = 0; i < a.size(); ++i) {
	    result += std::abs(a[i]-b[i]);
    }
    return result/a.size();
}

int main(int argc, char **argv) {
	if (argc < 2) {
    		std::cout << "Usage: " << argv[0] << " <file>" << std::endl;
    		return 1;
  	}


	std::vector<double> dense_vec;
	if(argc == 3){
		dense_vec = load_dense_vec_from_file(argv[2]);
	}
	std::vector<double> original_res;
	if(argc == 4){
		original_res = load_dense_vec_from_file(argv[3]);
	}


  	COO coo;
	std::cout << "loading matrix" << std::endl;
  	if(!coo.load_from_file(argv[1])){
		return -1;
  	}else{
		std::cout << "matrix loaded with success!" << std::endl;
	}
	
	if(dense_vec.size() == 0){
		dense_vec = std::vector<double>(coo.getMatrix().rows,1);
	}



	TIMER_DEF;
	float error, cputime, gputime,error_original,gputime_event;

	std::cout << "CPU : Compute" << std::endl;
	TIMER_START;
	auto cpu_result = coo.cpu_compute(dense_vec);
	TIMER_STOP;

	cputime = TIMER_ELAPSED;

					    
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);
	int blk_size = std::min(128, props.maxThreadsPerBlock);
	int grd_size = (coo.getMatrix().nnz + blk_size - 1) / blk_size;
	
	
	cudaEvent_t start, stop; 
    	cudaEventCreate(&start); 
    	cudaEventCreate(&stop); 


	GPU_COO_Pointers gpu_pointers = coo.gpu_prep(dense_vec.data());
	TIMER_START;
	std::cout << "GPU : Compute" << std::endl;
    	cudaEventRecord(start); 
	coo.gpu_compute(gpu_pointers, grd_size, blk_size);
	TIMER_STOP;
	gputime = TIMER_ELAPSED;
    	cudaEventRecord(stop); 
    	cudaEventSynchronize(stop); 
    
    	cudaEventElapsedTime(&gputime_event, start, stop); 

    	cudaEventDestroy(start); 
    	cudaEventDestroy(stop);


	auto gpu_result = coo.gpu_retrive(gpu_pointers);
	coo.gpu_free(gpu_pointers);

	std::cout << "Comparing" << std::endl;

	bool result = compare_vectors(cpu_result, gpu_result);
	error = diff_vector(cpu_result, gpu_result);
	error_original =  diff_vector(gpu_result, original_res);

	if(result){
		std::cout << "2 Vector are similar" << std::endl;
	}else{
		std::cout << "2 Vector not are similar!!!" << std::endl;
	}



    // ============================================ Print the results ============================================

    printf("================================== Times and results of my code ==================================\n");
    
    printf("Error between Original and GPU is %lf\n", error_original);
    printf("Error between CPU and GPU is %lf\n", error);
    printf("\nVector len = %d, CPU time = %5.3f\n", coo.getMatrix().nnz, cputime);
    printf("\nblk_size = %d, grd_size = %d, GPU time (gettimeofday): %5.3f sec\n", blk_size, grd_size, gputime);
    printf("\nblk_size = %d, grd_size = %d, GPU time (event): %5.3f ms\n", blk_size, grd_size, gputime_event);
}
