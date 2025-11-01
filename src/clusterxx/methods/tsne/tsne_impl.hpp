#ifndef CLUSTERXX_METHODS_TSNE_IMPL_HPP
#define CLUSTERXX_METHODS_TSNE_IMPL_HPP

#include "clusterxx/data_structures/kd_tree/kd_tree.hpp"
#include "tsne.hpp"

#include <float.h>
#include <random>

#include <chrono>

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
    const arma::mat &features) {
    arma::mat p_ji;
    arma::mat pairwise_dists;
    p_ji.resize(features.n_rows, features.n_rows);
    pairwise_dists = metric(features);
    for (size_t i = 0; i < features.n_rows; i++) {
        double beta = __compute_beta(pairwise_dists, __perplexity, i);

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
    auto t1 = std::chrono::high_resolution_clock::now();
    arma::mat pairwise_dists = metric(Y);
    auto t2 = std::chrono::high_resolution_clock::now();

    auto t3 = std::chrono::high_resolution_clock::now();
    arma::mat q_ij = 1.0 / (1.0 + pairwise_dists);
    q_ij.diag().zeros();
    auto t4 = std::chrono::high_resolution_clock::now();

    auto t5 = std::chrono::high_resolution_clock::now();
    double total_sum = arma::accu(q_ij);
    auto t6 = std::chrono::high_resolution_clock::now();

    q_ij /= total_sum;

    auto compute_metric = std::chrono::duration_cast<std::chrono::milliseconds>(
        t2 - t1
    ).count();

    auto compute_q_ij = std::chrono::duration_cast<std::chrono::milliseconds>(
        t4 - t3
    ).count();

    auto compute_sum = std::chrono::duration_cast<std::chrono::milliseconds>(
        t6 - t5
    ).count();

    std::cout << '[' << compute_metric << " - " << compute_q_ij << " - " << compute_sum << "]\n\n";

    return std::make_pair(q_ij, pairwise_dists);
}

template <typename Metric>
arma::mat clusterxx::TSNE<Metric>::__kullback_leibler_gradient(
    const __gradient_data &data) {
    arma::mat gradient;
    if (__method == "exact") {
        assert(data.pairwise_affinities.n_rows ==
               data.low_dim_affinities.n_rows);
        assert(data.pairwise_affinities.n_cols ==
               data.low_dim_affinities.n_cols);

        arma::mat F = 4.0 * (__early_exaggeration * data.pairwise_affinities -
                             data.low_dim_affinities);
        F /= (1.0 + data.pairwise_dists);
        F.diag().zeros();
        gradient = arma::diagmat(arma::sum(F, 1)) * data.low_dim_features -
                   F * data.low_dim_features;
    } else { // barnes-hut
    }
    return gradient;
}

template <typename Metric>
void clusterxx::TSNE<Metric>::__fit(const arma::mat &X) {
    assert(!X.empty());
    assert(__n_components < X.n_cols);
    assert(__perplexity < X.n_rows);

    __features = X;
    arma::mat p_ji = __compute_pairwise_affinities(X);
    arma::mat symmetrized;
    symmetrized = (p_ji + p_ji.t()) / (2.0 * X.n_rows);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1e-4);

    arma::mat Y(X.n_rows, __n_components);
    for (size_t i = 0; i < Y.n_rows; i++) {
        for (size_t j = 0; j < __n_components; j++) {
            Y(i, j) = dist(gen);
        }
    }
    __gradient_data g_data = {.pairwise_affinities = symmetrized,
                              .low_dim_features = Y};

    arma::mat Y_inc = arma::zeros<arma::mat>(Y.n_rows, Y.n_cols);
    double best_loss = std::numeric_limits<double>::infinity();
    uint32_t n_iter_no_progress = 0;
    for (int i = 0; i < __max_iter; i++) {
        if (i == static_cast<int>(0.25 * __max_iter)) [[unlikely]] {
            // increase momentum and set early exaggeration to 1
            // after 1/4 of the iterations
            __momentum = 0.8;
            __early_exaggeration = 1.0;
        }

        // auto t1 = std::chrono::high_resolution_clock::now();
        auto [q_ij, sol_pairwise_dists] = __compute_low_dim_affinities(Y);
        // auto t2 = std::chrono::high_resolution_clock::now();
        __gradient_data g_data = {.pairwise_affinities = symmetrized,
                                  .low_dim_affinities = q_ij,
                                  .low_dim_features = Y,
                                  .pairwise_dists = sol_pairwise_dists};

        // auto t3 = std::chrono::high_resolution_clock::now();
        arma::mat gradients = __kullback_leibler_gradient(g_data);
        // auto t4 = std::chrono::high_resolution_clock::now();

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

        // auto t5 = std::chrono::high_resolution_clock::now();
        Y_inc = (__momentum * Y_inc) - (__learning_rate * gradients);
        Y += Y_inc;
        // auto t6 = std::chrono::high_resolution_clock::now();

        // auto low_dim_affinities_computation = std::chrono::duration_cast<std::chrono::milliseconds>(
        //     t2 - t1
        // ).count();
        // auto gradient_computation = std::chrono::duration_cast<std::chrono::milliseconds>(
        //     t4 - t3
        // ).count();
        // auto update_solution_computation = std::chrono::duration_cast<std::chrono::milliseconds>(
        //     t6 - t5
        // ).count();

        // std::cout << '[' << low_dim_affinities_computation << " - " << gradient_computation << " - " << update_solution_computation << '\n';
    }

    __out_features = Y;
    __shape.first = Y.n_rows;
    __shape.second = __n_components;
}

template <typename Metric>
clusterxx::TSNE<Metric>::TSNE(
    const uint16_t n_components, const double perplexity,
    const double learning_rate, const double early_exaggeration,
    const uint32_t max_iter, const double min_grad_norm,
    const uint32_t n_iter_without_progress, const std::string method)
    : __n_components(n_components), __perplexity(perplexity),
      __learning_rate(learning_rate), __early_exaggeration(early_exaggeration),
      __max_iter(max_iter), __min_grad_norm(min_grad_norm),
      __n_iter_without_progress(n_iter_without_progress), __method(method) {
    assert(n_components > 0);
    assert(perplexity > 0);
    assert(learning_rate > 0.0);
    assert(early_exaggeration >= 1.0);
    assert(max_iter >= 20);
    assert(min_grad_norm > 0.0);
    assert(n_iter_without_progress > 0);
    assert(method == "barnes_hut" || method == "exact");
}

template <typename Metric>
void clusterxx::TSNE<Metric>::fit(const arma::mat &X) {
    __fit(X);
}

template <typename Metric>
arma::mat clusterxx::TSNE<Metric>::fit_transform(const arma::mat &X) {
    __fit(X);
    return __out_features;
}

#endif
