#ifndef CLUSTERXX_TSNE_UTILS_HPP
#define CLUSTERXX_TSNE_UTILS_HPP

#include <cassert>
#include <cmath>
#include <math.h>
#include <vector>

auto vector_diff = [](std::vector<double> a, std::vector<double> b) -> double {
    assert(a.size() == b.size());
    double res = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        res += (a[i] - b[i]) * (a[i] - b[i]);
    }

    return res;
};

double compute_sigma(const std::vector<double> &distances,
                     double target_perplexity, double tolerance = 1e-5,
                     int max_iter = 100) {
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

std::vector<std::vector<double>>
compute_pairwise_affinities(std::vector<std::vector<double>> features,
                            double perplexity) {
    std::vector<std::vector<double>> p_ji(features.size(),
                                          std::vector<double>(features.size()));
    for (size_t i = 0; i < features.size(); i++) {
        std::vector<double> distances;
        for (size_t j = 0; j < features.size(); j++) {
            if (i != j) {
                distances.push_back(vector_diff(features[i], features[j]));
            }
        }

        double sigma = compute_sigma(distances, perplexity);

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

std::vector<std::vector<double>>
compute_low_dim_affinities(const std::vector<std::vector<double>> Y) {
    std::vector<std::vector<double>> q_ij(Y.size(),
                                          std::vector<double>(Y.size()));
    std::vector<double> distances;
    for (size_t i = 0; i < Y.size(); i++) {
        for (size_t j = 0; j < Y.size(); j++) {
            if (i != j) {
                distances.push_back(vector_diff(Y[i], Y[j]));
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

std::vector<std::vector<double>>
kullback_leibler_gradient(std::vector<std::vector<double>> pairwise_affinities,
                          std::vector<std::vector<double>> low_dim_affinities,
                          std::vector<std::vector<double>> low_dim_features) {
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
                    (1.0 +
                     vector_diff(low_dim_features[i], low_dim_features[j]));

                for (size_t k = 0; k < low_dim_features[0].size(); k++) {
                    gradient[i][k] += factor * (low_dim_features[i][k] -
                                                low_dim_features[j][k]);
                }
            }
        }
    }

    return gradient;
}

#endif
