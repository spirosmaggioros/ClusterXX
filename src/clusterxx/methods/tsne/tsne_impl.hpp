#ifndef CLUSTERXX_METHODS_TSNE_IMPL_HPP
#define CLUSTERXX_METHODS_TSNE_IMPL_HPP

#include "tsne.hpp"
#include <random>

template <typename Metric>
double
clusterxx::TSNE<Metric>::__compute_sigma(const arma::mat &distances,
                                         double target_perplexity, int iter,
                                         double tolerance, int max_iter) {
    double sigma_lo = 1e-10, sigma_hi = 1e10;
    double sigma = 1.0;

    while (max_iter--) {
        double sum_exp = 0.0, sum_exp_tot = 0.0;
        for (size_t j = 0; j < distances.row(iter).n_rows; j++) {
            double exp = std::exp(-1.0 * distances(iter, j) / (2.0 * sigma * sigma));
            sum_exp += exp;
            sum_exp_tot += distances(iter, j) * exp;
        }

        if (sum_exp == 0) {
            sigma_lo = sigma;
            sigma = (sigma + sigma_hi) / 2.0;
            continue;
        }

        double entropy =
            std::log(sum_exp) + sum_exp_tot / (sum_exp * 2 * sigma * sigma);
        double perpl = std::pow(2, entropy);

        if (abs(perpl - target_perplexity) < tolerance) {
            break;
        }

        if (perpl > target_perplexity) {
            sigma_hi = sigma;
        } else {
            sigma_lo = sigma;
        }
        sigma = (sigma_lo + sigma_hi) / 2.0;
    }

    return sigma;
}

template <typename Metric>
arma::mat clusterxx::TSNE<Metric>::__compute_pairwise_affinities(
    const arma::mat &features, double perplexity) {
    arma::mat p_ji(features.n_rows, features.n_rows);
    arma::mat pairwise_dists = metric(features, arma::mat());
    for (size_t i = 0; i < features.n_rows; i++) {
        std::cout << "i am at iteration: " << i << " / " << features.size() << '\n';
        double sigma = __compute_sigma(pairwise_dists, perplexity, i);

        double sum = 0.0;
        size_t dist_idx = 0;
        for (size_t j = 0; j < features.n_rows; j++) {
            if (i == j) [[unlikely]] {
                p_ji(i, j) = 0.0;
            } else {
                p_ji(i, j) = std::exp(-1.0 * pairwise_dists(i, dist_idx) /
                                      (2.0 * sigma * sigma));
                sum += p_ji(i, j);
                dist_idx++;
            }
        }
        if (sum > 0) {
            for (auto &p: p_ji.row(i)) {
                p /= sum;
            }
        }
    }

    return p_ji;
}

template <typename Metric>
arma::mat clusterxx::TSNE<Metric>::__compute_low_dim_affinities(const arma::mat &Y) {
    arma::mat q_ij(Y.n_rows, Y.n_rows);
    arma::mat pairwise_dists = metric(Y, arma::mat());
    for (size_t i = 0; i < Y.n_rows; i++) {
        size_t dist_idx = 0;
        double sum = 0.0;
        for (size_t j = 0; j < Y.n_rows; j++) {
            if (i == j) [[unlikely]] {
                q_ij(i, j) = 0.0;
            } else {
                q_ij(i, j) = 1.0 / (1.0 + pairwise_dists(i, dist_idx));
                sum += q_ij(i, j);
                dist_idx++;
            }
        }

        if (sum > 0) {
            for (auto &q: q_ij.row(i)) {
                q /= sum;
            }
        }
    }

    return q_ij;
}

template <typename Metric>
arma::mat clusterxx::TSNE<Metric>::__kullback_leibler_gradient(
    const arma::mat &pairwise_affinities,
    const arma::mat &low_dim_affinities,
    const arma::mat &low_dim_features) {
    assert(pairwise_affinities.n_rows == low_dim_affinities.n_rows);
    assert(pairwise_affinities.n_cols == low_dim_affinities.n_cols);

    arma::mat pairwise_dists = metric(low_dim_features, arma::mat());
    arma::mat gradient(pairwise_affinities.n_rows, low_dim_features.n_cols);

    for (size_t i = 0; i < pairwise_affinities.n_rows; i++) {
        for (size_t j = 0; j < pairwise_affinities.n_cols; j++) {
            if (i != j) [[likely]] {
                double factor =
                    4.0 *
                    ((__early_exaggeration * pairwise_affinities(i, j)) -
                     low_dim_affinities(i, j)) /
                    (1.0 + pairwise_dists(i, j));

                for (size_t k = 0; k < low_dim_features.n_cols; k++) {
                    gradient(i, k) += factor * (low_dim_features(i, k) -
                                                low_dim_features(j, k));
                }
            }
        }
    }

    return gradient;
}

template <typename Metric>
void clusterxx::TSNE<Metric>::__fit(const arma::mat &X) {
    assert(!X.empty());
    assert(__n_components < X.n_cols);
    assert(__perplexity < X.n_rows);

    // compute pairwise affinities
    arma::mat p_ji = __compute_pairwise_affinities(X, __perplexity);
    arma::mat symmetrized = p_ji;
    for (size_t i = 0; i < p_ji.n_rows; i++) {
        for (size_t j = 0; j < p_ji.n_cols; j++) {
            symmetrized(i, j) = (p_ji(i, j) + p_ji(j, i)) / (2.0 * X.n_rows);
        }
    }

    // sample initial solution from N(0, 1e-4)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1e-4);

    arma::mat solution(X.n_rows, __n_components);
    for (size_t i = 0; i < solution.n_rows; i++) {
        for (size_t j = 0; j < __n_components; j++) {
            solution(i, j) = dist(gen);
        }
    }

    std::vector<arma::mat> solution_hist;

    for (int i = 0; i < __max_iter; i++) {
        std::cout << " im at iter: " << i << '\n';
        if (i == static_cast<int>(__max_iter / 20)) {
            // end early exaggeration after __max_iter / 20(default, 20
            // iterations)
            __early_exaggeration = 1.0;
        }
        if (i == static_cast<int>(0.25 * __max_iter)) [[unlikely]] {
            // increase momentum after 1/4 of the iterations
            __momentum = 0.8;
        }

        // compute low dimensional affinities(q_ij)
        arma::mat q_ij = __compute_low_dim_affinities(solution);

        // compute gradient
        arma::mat gradients = __kullback_leibler_gradient(symmetrized, q_ij, solution);

        // update solution
        for (size_t j = 0; j < solution.n_rows; j++) {
            for (size_t k = 0; k < __n_components; k++) {

                double momentum_factor = 0.0, previous_factor = 0.0;
                if (i > 0) [[likely]] {
                    previous_factor = solution_hist.back()(j, k);
                }
                if (i > 1) [[likely]] {
                    momentum_factor =
                        solution_hist.back()(j, k) -
                        solution_hist[solution_hist.size() - 2](j, k);
                }
                solution(j, k) = previous_factor +
                                 __learning_rate * gradients(j, k) +
                                 __momentum * momentum_factor;
            }
        }
        solution_hist.push_back(solution);
    }

    __features = solution;
    __shape.first = solution.n_rows;
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
