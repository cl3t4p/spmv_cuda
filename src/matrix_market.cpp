#include "matrix_market.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <thread>
#include <vector>


std::vector<std::string> split_line(std::string line){
	std::istringstream iss(line);
	std::vector<std::string> result;
	std::string val;
	while(iss >> val){
		result.push_back(val);
	}

	return result;
}

bool COO::load_from_file(std::string path){
	auto ifs_mtx = std::ifstream(path);
	if(!ifs_mtx.is_open()){
		std::cerr << "file "<< path << " does not exists!" << std::endl;
		return false;
	}
	std::string line;
	while(std::getline(ifs_mtx,line)){
		if(line.at(0) != '%'){
			break;
		}
	}

	auto coo_info = split_line(line);
	this->rows = std::stoull(coo_info[0]);
	this->cols = std::stoull(coo_info[1]);
	this->nnz = std::stoull(coo_info[2]);


	while(std::getline(ifs_mtx,line)){
		COO_Entry entry;
		auto entry_info = split_line(line);
		entry.row = std::stoull(entry_info[0]);
		entry.col = std::stoull(entry_info[1]);
		entry.val = std::stod(entry_info[2]);
		this->entries.push_back(entry);
	}

	if(this->entries.size() != this->nnz){
		std::cerr << "nnz "<< nnz << " size does not match matrix" << this->entries.size() << "!" << std::endl;
		return false;
	}
	return true;
}

double COO::compute_cell(uint32_t row,std::vector<double> dense_vec){
	double result = 0;
	for (COO_Entry entry : entries){
		if(entry.row == row){
			result += entry.val * dense_vec[entry.col-1];
		}
	}
	return result;
}



std::vector<double> COO::multiply_cpu(std::vector<double> dense_vec,uint8_t n_process){
	std::vector<double> result(dense_vec.size());
	for(uint32_t x = 0;x < dense_vec.size();x++){
		result[x] = compute_cell(x+1, dense_vec);
	}
	return result;
}






