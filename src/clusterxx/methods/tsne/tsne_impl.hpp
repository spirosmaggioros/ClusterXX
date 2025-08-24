#ifndef CLUSTERXX_METHODS_TSNE_IMPL_HPP
#define CLUSTERXX_METHODS_TSNE_IMPL_HPP

#include "tsne.hpp"
#include <random>

template <typename Metric>
double
clusterxx::TSNE<Metric>::__compute_sigma(const std::vector<double> &distances,
                                         double target_perplexity,
                                         double tolerance, int max_iter) {
    double sigma_lo = 1e-20, sigma_hi = 1e20;
    double sigma = 1.0;

    while (max_iter--) {
        double sum_exp = 0.0, sum_exp_tot = 0.0;
        for (size_t j = 0; j < distances.size(); j++) {
            double exp = std::exp(-1.0 * distances[j] / (2.0 * sigma * sigma));
            sum_exp += exp;
            sum_exp_tot += distances[j] * exp;
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
std::vector<std::vector<double>>
clusterxx::TSNE<Metric>::__compute_pairwise_affinities(
    const std::vector<std::vector<double>> &features, double perplexity) {
    std::vector<std::vector<double>> p_ji(features.size(),
                                          std::vector<double>(features.size()));
    for (size_t i = 0; i < features.size(); i++) {
        std::vector<double> distances;
        for (size_t j = 0; j < features.size(); j++) {
            if (i != j) {
                distances.push_back(metric(features[i], features[j]));
            }
        }

        double sigma = __compute_sigma(distances, perplexity);

        double sum = 0.0;
        size_t dist_idx = 0;
        for (size_t j = 0; j < features.size(); j++) {
            if (i == j) [[unlikely]] {
                p_ji[i][j] = 0.0;
            } else {
                p_ji[i][j] = std::exp(-1.0 * distances[dist_idx] /
                                      (2.0 * sigma * sigma));
                sum += p_ji[i][j];
                dist_idx++;
            }
        }

        for (size_t j = 0; j < features.size(); j++) {
            if (sum > 0) {
                p_ji[i][j] /= sum;
            }
        }
    }

    return p_ji;
}

template <typename Metric>
std::vector<std::vector<double>>
clusterxx::TSNE<Metric>::__compute_low_dim_affinities(
    const std::vector<std::vector<double>> &Y) {
    std::vector<std::vector<double>> q_ij(Y.size(),
                                          std::vector<double>(Y.size()));
    std::vector<double> distances;
    for (size_t i = 0; i < Y.size(); i++) {
        for (size_t j = 0; j < Y.size(); j++) {
            if (i != j) {
                distances.push_back(metric(Y[i], Y[j]));
            }
        }

        size_t dist_idx = 0;
        double sum = 0.0;
        for (size_t j = 0; j < Y.size(); j++) {
            if (i == j) [[unlikely]] {
                q_ij[i][j] = 0.0;
            } else {
                q_ij[i][j] = 1.0 / (1.0 + distances[dist_idx]);
                sum += q_ij[i][j];
                dist_idx++;
            }
        }

        for (size_t j = 0; j < Y.size(); j++) {
            if (sum > 0) {
                q_ij[i][j] /= sum;
            }
        }
    }

    return q_ij;
}

template <typename Metric>
std::vector<std::vector<double>>
clusterxx::TSNE<Metric>::__kullback_leibler_gradient(
    const std::vector<std::vector<double>> &pairwise_affinities,
    const std::vector<std::vector<double>> &low_dim_affinities,
    const std::vector<std::vector<double>> &low_dim_features) {
    assert(pairwise_affinities.size() == low_dim_affinities.size());
    assert(pairwise_affinities[0].size() == low_dim_affinities[0].size());

    std::vector<std::vector<double>> gradient(
        pairwise_affinities.size(),
        std::vector<double>(low_dim_features[0].size()));

    for (size_t i = 0; i < pairwise_affinities.size(); i++) {
        for (size_t j = 0; j < pairwise_affinities[0].size(); j++) {
            if (i != j) {
                double factor =
                    4.0 *
                    (pairwise_affinities[i][j] - low_dim_affinities[i][j]) /
                    (1.0 + metric(low_dim_features[i], low_dim_features[j]));

                for (size_t k = 0; k < low_dim_features[0].size(); k++) {
                    gradient[i][k] += factor * (low_dim_features[i][k] -
                                                low_dim_features[j][k]);
                }
            }
        }
    }

    return gradient;
}

template <typename Metric>
void clusterxx::TSNE<Metric>::__fit(const std::vector<std::vector<double>> &X) {
    assert(!X.empty());
    assert(__n_components < X[0].size());
    assert(__perplexity < X.size());

    // compute pairwise affinities
    std::vector<std::vector<double>> p_ji =
        __compute_pairwise_affinities(X, __perplexity);
    std::vector<std::vector<double>> symmetrized = p_ji;
    for (size_t i = 0; i < p_ji.size(); i++) {
        for (size_t j = 0; j < p_ji[0].size(); j++) {
            symmetrized[i][j] = (p_ji[i][j] + p_ji[j][i]) / (2.0 * X.size());
        }
    }

    // sample initial solution from N(0, 1e-4)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1e-4);

    std::vector<std::vector<double>> solution(
        X.size(), std::vector<double>(__n_components));
    for (size_t i = 0; i < solution.size(); i++) {
        for (size_t j = 0; j < __n_components; j++) {
            solution[i][j] = dist(gen);
        }
    }

    std::vector<std::vector<std::vector<double>>> solution_hist;

    for (int i = 0; i < __max_iter; i++) {

        if (i == static_cast<int>(0.25 * __max_iter)) [[unlikely]] {
            // increase momentum after 1/4 of the iterations
            __momentum = 0.8;
        }

        // compute low dimensional affinities(q_ij)
        std::vector<std::vector<double>> q_ij =
            __compute_low_dim_affinities(solution);

        // compute gradient
        std::vector<std::vector<double>> gradients =
            __kullback_leibler_gradient(symmetrized, q_ij, solution);

        // update solution
        for (size_t j = 0; j < solution.size(); j++) {
            for (size_t k = 0; k < solution[j].size(); k++) {

                double momentum_factor = 0.0, previous_factor = 0.0;
                if (i > 0) [[likely]] {
                    previous_factor = solution_hist.back()[j][k];
                }
                if (i > 1) [[likely]] {
                    momentum_factor =
                        solution_hist.back()[j][k] -
                        solution_hist[solution_hist.size() - 2][j][k];
                }
                solution[j][k] = previous_factor +
                                 __learning_rate * gradients[j][k] +
                                 __momentum * momentum_factor;
            }
        }
        solution_hist.push_back(solution);
    }

    for (auto &v : solution) {
        __features.push_back(v);
    }
    __shape.first = solution.size();
    __shape.second = __n_components;
}

template <typename Metric>
void clusterxx::TSNE<Metric>::fit(const std::vector<std::vector<double>> &X) {
    __fit(X);
}

template <typename Metric>
std::vector<std::vector<double>> clusterxx::TSNE<Metric>::fit_transform(
    const std::vector<std::vector<double>> &X) {
    __fit(X);
    return __features;
}

template <typename Metric>
std::pair<int, int> clusterxx::TSNE<Metric>::get_shape() const {
    return __shape;
}

template <typename Metric>
std::vector<std::vector<double>> clusterxx::TSNE<Metric>::get_features() const {
    return __features;
}

#endif
