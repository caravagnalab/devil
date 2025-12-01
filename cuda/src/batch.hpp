#include <Eigen/Dense>
/*
Eigen::MatrixXf beta_fit_gpu_external(Eigen::MatrixXf Y_host, Eigen::MatrixXf
X_host, Eigen::MatrixXf mu_beta_host, Eigen::MatrixXf offset_host,
Eigen::VectorXf k_host, int max_iter, float eps);
*/

struct BatchResult {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> beta;
    Eigen::VectorXf k;
    Eigen::VectorXf theta;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> beta_init;  // Debug: initial beta values
};

BatchResult
beta_fit_gpu_external(
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> const &
        Y_host,
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> const &
        X_host,
    Eigen::VectorXf const & offset_host,
    int max_iter, float eps,int batch_size,std::vector<int>& iter, bool TEST = false);
