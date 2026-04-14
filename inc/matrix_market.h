//
// Created by cl3t4p on 11/04/2026.
//

#ifndef SPMV_MATRIX_MARKET_H
#define SPMV_MATRIX_MARKET_H
#include <cstdint>
#include <sys/types.h>
#include <string>
#include <vector>




typedef struct {
    u_int32_t rows;
    u_int32_t cols;
    u_int32_t nnz;
    u_int32_t *row_p;
    u_int32_t *col_p;
    double *val_p;
}COO_Matrix;


typedef struct {
	double *result;
	double *dense_vec;
	COO_Matrix matrix;
}GPU_COO_Pointers;

class COO{
	private:
		COO_Matrix matrix;
	public:
		~COO();
		
		GPU_COO_Pointers gpu_prep(double *dense_vec);
		void gpu_compute(const GPU_COO_Pointers pointers,uint grid_size,uint blk_size);
		const std::vector<double> gpu_retrive(GPU_COO_Pointers pointers);
		void gpu_free(GPU_COO_Pointers pointers);
		bool load_from_file(std::string path);
		std::vector<double> cpu_compute(std::vector<double> dense_vec);
		uint32_t getRows(){
			return matrix.rows;
		}
		uint32_t getCols(){
			return matrix.rows;
		}
		uint32_t getNnz(){
			return matrix.rows;
		}
		COO_Matrix getMatrix(){
			return matrix;
		}
};

typedef struct{
	uint32_t rows;
	uint32_t cols;
	uint32_t nnz;
	std::vector<uint32_t> rowPtrs;
	std::vector<uint32_t> colIdx;
	std::vector<double> val;
}CSR_Matrix;

class CSR{
	private:
		CSR_Matrix matrix;
	public:
		void load_from_coo(const COO &coo);
		std::vector<double> multiply_gpu(std::vector<double> dense_vec);
		CSR_Matrix* getMatrix(){
			return &matrix;
		}
};




std::vector<double> load_dense_vec_from_file(std::string path);
#endif //SPMV_MATRIX_MARKET_H
