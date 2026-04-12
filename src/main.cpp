#include <iostream>
#include <ostream>
#include <vector>

#include "matrix_market.h"

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <file>" << std::endl;
    return 1;
  }

  COO coo;
  if(!coo.load_from_file(argv[1])){
	  return -1;
  }
  std::cout << "loaded file : " << argv[1] << std::endl;


  std::vector<double> dense_vec(coo.getRows(),1);

  auto result = coo.multiply_cpu(dense_vec, 10);

  for (size_t i = 0; i < result.size(); ++i) {
    std::cout << "result[" << i << "] = " << result[i] << std::endl;
  }
}
