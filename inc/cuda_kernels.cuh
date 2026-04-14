#include "matrix_market.h"


__global__ void spmv_coo_kernel(COO_Matrix matrix,double *dense_vec,double *result);
__global__ void spmv_csr_kernel(CSR_Matrix matrix,double *dense_vec,double *result);
