#include "cuda_kernels.cuh"
#include <device_atomic_functions.h>



__global__ void spmv_coo_kernel(COO_Matrix matrix,double *dense_vec,double *result){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i< matrix.nnz){
		// FIX Matrix start on 1
		int row = matrix.row_p[i]-1;
		int col = matrix.col_p[i]-1;
		double val = matrix.val_p[i];
		atomicAdd(&result[row],val* dense_vec[col]);
	}
}
__global__ void spmv_csr_kernel(CSR_Matrix matrix,double *dense_vec,double *result){

}
