#include "gp3d.h"
#include <cmath>
#include <Eigen/Eigenvalues>

using namespace Eigen;

static double softplus(double x) {
    return std::log1p(std::exp(-std::abs(x))) + std::max(x, 0.0);
}

GP3D::GP3D(const MatrixXd& X_, const VectorXd& y_, const GPConfig& cfg)
    : X(X_), y(y_), config(cfg), N(X_.rows()) {
}

double GP3D::kernel(int i, int j, const VectorXd& p) const {
    double l1 = softplus(p[0]);
    double l2 = softplus(p[1]);
    double l3 = softplus(p[2]);
    double sf2 = softplus(p[3]);

    double dx = X(i, 0) - X(j, 0);
    double dy = X(i, 1) - X(j, 1);
    double dz = X(i, 2) - X(j, 2);

    double r2 =
        (dx * dx) / (l1 * l1) +
        (dy * dy) / (l2 * l2) +
        (dz * dz) / (l3 * l3);

    return sf2 * std::exp(-0.5 * r2);
}

MatrixXd GP3D::computeK(const VectorXd& p) const {
    MatrixXd K(N, N);

    for (int i = 0; i < N; ++i) {
        for (int j = i; j < N; ++j) {
            double v = kernel(i, j, p);
            K(i, j) = v;
            K(j, i) = v;
        }
    }

    double sn2 = softplus(p[4]);
    K.diagonal().array() += sn2 + config.jitter;

    return K;
}

double GP3D::operator()(const VectorXd& p, VectorXd& grad) {
    VectorXd pp = p;

    for (int i = 0; i < pp.size(); ++i)
        pp[i] = std::max(std::min(pp[i], 10.0), -10.0);

    MatrixXd K = computeK(pp);

    LDLT<MatrixXd> ldlt(K);

    if (ldlt.info() != Success)
        return 1e50;

    VectorXd alpha_loc = ldlt.solve(y);

    double logdet = ldlt.vectorD().array().max(1e-12).log().sum();

    double ll =
        -0.5 * y.dot(alpha_loc) -
        0.5 * logdet -
        0.5 * N * std::log(2.0 * M_PI);

    MatrixXd K_inv = ldlt.solve(MatrixXd::Identity(N, N));
    MatrixXd A = alpha_loc * alpha_loc.transpose() - K_inv;

    grad.setZero(5);

    for (int d = 0; d < 3; ++d) {
        MatrixXd dK(N, N);

        double ld = softplus(pp[d]);
        double ld2 = ld * ld;

        for (int i = 0; i < N; ++i) {
            for (int j = i; j < N; ++j) {
                double kij = kernel(i, j, pp);
                double diff = X(i, d) - X(j, d);
                double val = kij * (diff * diff) / ld2;
                dK(i, j) = val;
                dK(j, i) = val;
            }
        }

        grad[d] = 0.5 * (A.cwiseProduct(dK)).sum();
    }

    {
        MatrixXd dK(N, N);

        for (int i = 0; i < N; ++i) {
            for (int j = i; j < N; ++j) {
                double kij = kernel(i, j, pp);
                double val = 2.0 * kij;
                dK(i, j) = val;
                dK(j, i) = val;
            }
        }

        grad[3] = 0.5 * (A.cwiseProduct(dK)).sum();
    }

    {
        double sn2 = softplus(pp[4]);
        MatrixXd dK = 2.0 * sn2 * MatrixXd::Identity(N, N);
        grad[4] = 0.5 * (A.cwiseProduct(dK)).sum();
    }

    if (!std::isfinite(ll))
        return 1e50;

    for (int i = 0; i < grad.size(); ++i)
        if (!std::isfinite(grad[i]))
            grad[i] = 0.0;

    grad *= -1.0;
    return -ll;
}

void GP3D::fit(const VectorXd& p) {
    trained_params = p;
    MatrixXd K = computeK(p);
    LDLT<MatrixXd> ldlt(K);
    if (ldlt.info() != Success) return;
    L = ldlt.matrixL();
    alpha = ldlt.solve(y);
}

VectorXd GP3D::compute_kstar(const Vector3d& x) const {
    VectorXd k(N);

    double l1 = softplus(trained_params[0]);
    double l2 = softplus(trained_params[1]);
    double l3 = softplus(trained_params[2]);
    double sf2 = softplus(trained_params[3]);

    for (int i = 0; i < N; ++i) {
        double dx = X(i, 0) - x[0];
        double dy = X(i, 1) - x[1];
        double dz = X(i, 2) - x[2];

        double r2 =
            (dx * dx) / (l1 * l1) +
            (dy * dy) / (l2 * l2) +
            (dz * dz) / (l3 * l3);

        k[i] = sf2 * std::exp(-0.5 * r2);
    }

    return k;
}

std::pair<double, double> GP3D::predict(const Vector3d& x) const {
    VectorXd k = compute_kstar(x);

    double mean = k.dot(alpha);

    VectorXd v = L.triangularView<Lower>().solve(k);

    double kxx = softplus(trained_params[3]);
    double var = kxx - v.squaredNorm();

    if (!std::isfinite(var)) var = 1e-12;
    if (var < 1e-12) var = 1e-12;

    return { mean, var };
}
