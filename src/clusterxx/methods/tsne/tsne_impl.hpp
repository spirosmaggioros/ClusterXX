#ifndef CLUSTERXX_METHODS_TSNE_IMPL_HPP
#define CLUSTERXX_METHODS_TSNE_IMPL_HPP

#include "tsne.hpp"
#include <float.h>
#include <random>

#include <chrono>
#include <iostream>

template <typename Metric>
double clusterxx::TSNE<Metric>::__compute_beta(const arma::mat &distances,
                                               double target_perplexity,
                                               int iter, double tolerance,
                                               int max_iter) {
    double beta = 1.0, min_beta = -DBL_MAX, max_beta = DBL_MAX;
    std::vector<double> curr_P(distances.n_cols);
    while (max_iter--) {
        for (size_t j = 0; j < distances.n_cols; j++) {
            if (static_cast<int>(j) == iter) [[unlikely]] {
                curr_P[j] = 0.0;
            } else {
                curr_P[j] = std::exp(-beta * distances(iter, j));
            }
        }
        double sum_P = std::accumulate(curr_P.begin(), curr_P.end(), 0.0);
        double H = 0.0;
        for (size_t j = 0; j < distances.n_cols; j++) {
            H += beta * (distances(iter, j) * curr_P[j]);
        }
        H = (H / sum_P) + std::log(sum_P);
        double diff = H - std::log(target_perplexity);
        if (std::abs(diff) < tolerance) {
            break;
        }

        if (diff > 0) {
            min_beta = beta;
            if (max_beta == DBL_MAX || max_beta == -DBL_MAX) {
                beta *= 2.0;
            } else {
                beta = (beta + max_beta) / 2.0;
            }
        } else {
            max_beta = beta;
            if (min_beta == -DBL_MAX || min_beta == DBL_MAX) {
                beta /= 2.0;
            } else {
                beta = (beta + min_beta) / 2.0;
            }
        }
    }

    return beta;
}

template <typename Metric>
arma::mat clusterxx::TSNE<Metric>::__compute_pairwise_affinities(
    const arma::mat &features, double perplexity) {
    arma::mat p_ji(features.n_rows, features.n_rows);
    arma::mat pairwise_dists = metric(features, arma::mat());
    for (size_t i = 0; i < features.n_rows; i++) {
        double beta = __compute_beta(pairwise_dists, perplexity, i);

        double sum = 0.0;
        for (size_t j = 0; j < features.n_rows; j++) {
            if (i == j) [[unlikely]] {
                p_ji(i, j) = 0.0;
            } else {
                p_ji(i, j) = std::exp(-beta * pairwise_dists(i, j));
                sum += p_ji(i, j);
            }
        }
        if (sum > 0) {
            p_ji.row(i) /= sum;
        }
    }

    return p_ji;
}

template <typename Metric>
std::pair<arma::mat, arma::mat>
clusterxx::TSNE<Metric>::__compute_low_dim_affinities(const arma::mat &Y) {
    arma::mat pairwise_dists = metric(Y);
    arma::mat q_ij = 1.0 / (1.0 + pairwise_dists);
    q_ij.diag().zeros();
    double total_sum = arma::accu(q_ij);
    q_ij /= total_sum;

    return std::make_pair(q_ij, pairwise_dists);
}

template <typename Metric>
arma::mat clusterxx::TSNE<Metric>::__kullback_leibler_gradient(const __gradient_data &data) {
    assert(data.pairwise_affinities.n_rows == data.low_dim_affinities.n_rows);
    assert(data.pairwise_affinities.n_cols == data.low_dim_affinities.n_cols);

    arma::mat F =
        4.0 * (__early_exaggeration * data.pairwise_affinities - data.low_dim_affinities);
    F /= (1.0 + data.pairwise_dists);
    F.diag().zeros();
    arma::mat gradient = arma::diagmat(arma::sum(F, 1)) * data.low_dim_features -
                         F * data.low_dim_features;
    return gradient;
}

template <typename Metric>
void clusterxx::TSNE<Metric>::__fit(const arma::mat &X) {
    assert(!X.empty());
    assert(__n_components < X.n_cols);
    assert(__perplexity < X.n_rows);

    // compute pairwise affinities
    arma::mat p_ji = __compute_pairwise_affinities(X, __perplexity);
    arma::mat symmetrized = (p_ji + p_ji.t()) / (2.0 * X.n_rows);

    // sample initial solution from N(0, 1e-4)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1e-4);

    arma::mat Y(X.n_rows, __n_components);
    for (size_t i = 0; i < Y.n_rows; i++) {
        for (size_t j = 0; j < __n_components; j++) {
            Y(i, j) = dist(gen);
        }
    }
    __gradient_data g_data = {
        .pairwise_affinities = symmetrized,
        .low_dim_features = Y
    };

    arma::mat Y_inc = arma::zeros<arma::mat>(Y.n_rows, Y.n_cols);
    double best_loss = std::numeric_limits<double>::infinity();
    int n_iter_no_progress = 0;
    for (int i = 0; i < __max_iter; i++) {
        if (i == static_cast<int>(0.25 * __max_iter)) [[unlikely]] {
            // increase momentum and set early exaggeration to 1
            // after 1/4 of the iterations
            __momentum = 0.8;
            __early_exaggeration = 1.0;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        // compute low dimensional affinities(q_ij)
        auto [q_ij, sol_pairwise_dists] = __compute_low_dim_affinities(Y);
        g_data.low_dim_affinities = q_ij;
        g_data.pairwise_dists = sol_pairwise_dists;
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_double = t2 - t1;
        std::cout << "__compute_low_dim_affinities took: " << ms_double << '\n';

        t1 = std::chrono::high_resolution_clock::now();
        // compute gradient
        arma::mat gradients = __kullback_leibler_gradient(g_data);
        t2 = std::chrono::high_resolution_clock::now();
        ms_double = t2 - t1;
        std::cout << "__kullback_leibler_gradient took: " << ms_double << '\n';
        double grad_norm = arma::norm(gradients, "fro");
        if (grad_norm < __min_grad_norm) {
            break;
        }

        if (i > static_cast<int>(0.25 * __max_iter) && i % 50 == 0) {
            double kl_loss =
                arma::accu((symmetrized + 1e-12) %
                           arma::log((symmetrized + 1e-12) / (q_ij + 1e-12)));
            if (kl_loss > best_loss) {
                n_iter_no_progress += 50;
            } else {
                n_iter_no_progress = 0;
                best_loss = kl_loss;
            }

            if (n_iter_no_progress >= __n_iter_without_progress) {
                break;
            }
        }

        t1 = std::chrono::high_resolution_clock::now();
        // update solution
        Y_inc = (__momentum * Y_inc) - (__learning_rate * gradients);
        Y += Y_inc;
        t2 = std::chrono::high_resolution_clock::now();
        ms_double = t2 - t1;
        std::cout << "updating solution took: " << ms_double << '\n';
    }

    __features = Y;
    __shape.first = Y.n_rows;
    __shape.second = __n_components;
}

template <typename Metric>
void clusterxx::TSNE<Metric>::fit(const arma::mat &X) {
    __fit(X);
}

template <typename Metric>
arma::mat clusterxx::TSNE<Metric>::fit_transform(const arma::mat &X) {
    __fit(X);
    return __features;
}

template <typename Metric>
std::pair<int, int> clusterxx::TSNE<Metric>::get_shape() const {
    return __shape;
}

template <typename Metric>
arma::mat clusterxx::TSNE<Metric>::get_features() const {
    return __features;
}

#endif
