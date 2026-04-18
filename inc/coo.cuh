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



__global__ void spmv_coo_kernel(const COO_Matrix &matrix, const double *dense_vec,double *result);


class COO{
	private:
		COO_Matrix matrix = {};
	public:
		~COO();
		
		GPU_COO_Pointers gpu_prep(const double *dense_vec);
		std::vector<double> gpu_retrive(const GPU_COO_Pointers &pointers);
		inline void gpu_compute(const GPU_COO_Pointers* pointers,const uint grid_size, const uint blk_size){
			spmv_coo_kernel<<<grid_size,blk_size>>>(pointers->matrix, pointers->dense_vec,pointers->result);
		}

		inline void free_result(const GPU_COO_Pointers* pointers){
			cudaMemset(pointers->result, 0.0, getRows() * sizeof(double));
		}

		static void gpu_free(const GPU_COO_Pointers &pointers);
		bool load_from_file(const std::string &path);
		std::vector<double> cpu_compute(const std::vector<double> &dense_vec);
		[[nodiscard]] uint32_t getRows() const{
			return matrix.rows;
		}
		[[nodiscard]] uint32_t getCols() const{
			return matrix.cols;
		}
		[[nodiscard]] uint32_t getNnz() const{
			return matrix.nnz;
		}
		[[nodiscard]] COO_Matrix getMatrix() const {
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
		CSR_Matrix matrix = {};
	public:
		void load_from_coo(const COO &coo);
		std::vector<double> multiply_gpu(std::vector<double> dense_vec);
		CSR_Matrix* getMatrix(){
			return &matrix;
		}
};

#endif //SPMV_MATRIX_MARKET_H
