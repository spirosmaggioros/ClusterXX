#ifndef CLUSTERXX_METRICS_UTILS_HPP
#define CLUSTERXX_METRICS_UTILS_HPP

#include <assert.h>
#include <math.h>
#include <vector>
#include <thread>
#include <armadillo>

#include <iostream>

template <typename Metric>
void __compute_partial(const std::vector<std::vector<double>> &X,
                       const std::vector<std::vector<double>> &Y,
                       std::vector<std::vector<double>> &dist_vec,
                       size_t range_from, size_t range_to) {
    Metric metric;
    if (Y.empty()) {
        for (size_t i = range_from; i <= range_to; i++) {
            for (size_t j = i; j < X.size(); j++) {
                double d = metric(X[i], X[j]);
                dist_vec[i][j] = d;
                dist_vec[j][i] = d;
            }
        }
    } else {
        for (size_t i = range_from; i <= range_to; i++) {
            for (size_t j = 0; j < Y.size(); j++) {
                dist_vec[i][j] = metric(X[i], Y[j]);
            }
        }
    }
}

template <typename Metric>
void __compute_full(const std::vector<std::vector<double>> &X,
                    const std::vector<std::vector<double>> &Y,
                    std::vector<std::vector<double>> &dist_vec) {
    Metric metric;

    if (Y.empty()) {
        for (size_t i = 0; i < X.size(); i++) {
            for (size_t j = i; j < X.size(); j++) {
                double d = metric(X[i], X[j]);
                dist_vec[i][j] = d;
                dist_vec[j][i] = d;
            }
        }
    } else {
        for (size_t i = 0; i < X.size(); i++) {
            for (size_t j = 0; j < Y.size(); j++) {
                dist_vec[i][j] = metric(X[i], Y[j]);
            }
        }
    }
}

template <typename Metric>
std::vector<std::vector<double>>
compute_pairwise_distances(const std::vector<std::vector<double>> &X,
                           const std::vector<std::vector<double>> &Y) {;
    std::vector<std::vector<double>> dist;
    if (Y.empty()) { // only compute pairwise distances for X
        dist.resize(X.size(), std::vector<double>(X.size(), 0.0));
    } else {
        assert(X[0].size() == Y[0].size());
        dist.resize(X.size(), std::vector<double>(Y.size()));
    }

    __compute_full<Metric>(X, Y, dist);
    return dist;
}

arma::mat vvec_to_mat(const std::vector<std::vector<double>> &X) {
    if (X.empty()) { return arma::mat(); }
    
    arma::mat _X(X.size(), X[0].size());
    for (size_t i = 0; i < X.size(); i++) {
        for (size_t j = 0; j < X[0].size(); j++) {
            _X(i, j) = X[i][j];
        }
    }

    return _X;
}

std::vector<std::vector<double>> arma_to_2dvec(const arma::mat &X) {
    if (X.empty()) { return {}; }

    std::vector<std::vector<double>> _X(X.n_rows, std::vector<double>(X.n_cols));

    for (size_t i = 0; i < X.n_rows; i++) {
        for (size_t j = 0; j < X.n_cols; j++) {
            _X[i][j] = X(i, j);
        }
    }

    return _X;
}

#endif
