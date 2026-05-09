#ifndef GP3D_H
#define GP3D_H

#define _USE_MATH_DEFINES

#include <Eigen/Dense>
#include <utility>

struct GPConfig {
    int dim;

    double sigma_f;
    double sigma_n;

    Eigen::VectorXd lengths;

    double jitter;

    int max_iterations;

    GPConfig(int d = 3)
        :
        dim(d),
        sigma_f(1.0),
        sigma_n(0.1),
        lengths(Eigen::VectorXd::Ones(d)),
        jitter(1e-6),
        max_iterations(100)
    {
    }
};

class GP3D {
public:
    GPConfig config;

    Eigen::MatrixXd X;
    Eigen::VectorXd y;

    int N;

    Eigen::VectorXd trained_params;

    Eigen::VectorXd alpha;
    Eigen::MatrixXd L;

    GP3D(
        const Eigen::MatrixXd& X_,
        const Eigen::VectorXd& y_,
        const GPConfig& cfg = GPConfig()
    );

    double operator()(
        const Eigen::VectorXd& p,
        Eigen::VectorXd& grad
        );

    void fit(const Eigen::VectorXd& p);

    std::pair<double, double> predict(
        const Eigen::Vector3d& x
    ) const;

private:
    double kernel(
        int i,
        int j,
        const Eigen::VectorXd& p
    ) const;

    Eigen::MatrixXd computeK(
        const Eigen::VectorXd& p
    ) const;

    Eigen::VectorXd compute_kstar(
        const Eigen::Vector3d& x
    ) const;
};

#endif