#include "batch.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <fstream>


std::vector<float> readDatFile(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    std::cerr << "Unable to open file " << filename << std::endl;
    return {};
  }
  // Get the file size
  file.seekg(0, std::ios::end);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  // Read the data
  std::vector<float> data(size / sizeof(float));
  if (file.read(reinterpret_cast<char *>(data.data()), size)) {
    std::cout << "Loading file " << filename << " Success,size: " << data.size() << std::endl;
    return data;
  } else {
    std::cerr << "Error reading file " << filename << std::endl;
    return {};
  }
}

int main() {

 auto X_hostv = readDatFile("../data/X.dat");
 auto Y_hostv = readDatFile("../data/Y.dat");
 auto offset_hostv = readDatFile("../data/off.dat");
 auto mu_beta_hostv = readDatFile("../data/mu_beta.dat");
 auto k_hostv = readDatFile("../data/K.dat");
 for (auto &x : k_hostv)
   x=1/x;
 int cells=1024;
 int genes=64;
 int features=2;

 Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>   Y_host(Y_hostv.data(), genes,cells); //
 Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>  X_host(X_hostv.data(), cells,features); //
 Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>  mu_beta_host(mu_beta_hostv.data(),genes,features); 
 Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>  offset_host(offset_hostv.data(),  genes,cells);
 Eigen::Map<Eigen::VectorXf>  k_host(k_hostv.data(), genes);

 auto Y_hostT = Y_host.transpose();
 auto X_hostT = X_host.transpose();
 auto mu_beta_hostT = mu_beta_host.transpose();
 auto offset_hostT = offset_host.transpose();
 Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Y = Y_hostT;
 Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> X = X_hostT;
 Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> mu_beta =  mu_beta_hostT;
 Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> offset = offset_hostT;

 std::cout << "Data loaded" << std::endl;
 std::vector<int> iterations(genes);
 auto result = beta_fit_gpu_external(Y, X, mu_beta, offset, k_host, 100, 1e-3,
                                     2, iterations);
 
 std::cout << "DONE \n";
 std::cout << result.transpose() << std::endl;
 for (int i : iterations) {
   std::cout << " " << i ;
 }
 std::cout << std::endl;

}
