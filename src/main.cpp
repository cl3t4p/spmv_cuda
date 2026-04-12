#include <iostream>
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

}
