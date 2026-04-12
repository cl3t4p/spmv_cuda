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
    u_int32_t row;
    u_int32_t col;
    double val;
}COO_Entry;

typedef struct {
    u_int32_t rows;
    u_int32_t cols;
    u_int32_t nnz;
    COO_Entry *entries;
}COO_Struct;


COO_Struct* read_matrix_market(const char *filename);


class COO{
	private:
		std::vector<COO_Entry> entries;
		uint32_t rows;
		uint32_t cols;
		uint32_t nnz;
		
	public:
		bool load_from_file(std::string path);
		std::vector<double> multiply_cpu(std::vector<double> dense_vec,uint8_t n_process);
		uint32_t getRows(){
			return rows;
		}
		uint32_t getCols(){
			return cols;
		}
		uint32_t getNnz(){
			return nnz;
		}
		std::vector<COO_Entry> getEntries(){
			return entries;
		}
		double compute_cell(uint32_t row,const std::vector<double> &dense_vec);
};


#endif //SPMV_MATRIX_MARKET_H
