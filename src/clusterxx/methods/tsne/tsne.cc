#include "tsne.hpp"
#include "utils.hpp"
#include <random>

template <typename Metric>
void clusterxx::TSNE<Metric>::__fit(const std::vector<std::vector<double>> &X) {
    assert(__n_components < X[0].size());
    assert(__perplexity < X.size());

    // compute pairwise affinities
    std::vector<std::vector<double>> p_ji =
        compute_pairwise_affinities(X, __perplexity, metric);
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
            compute_low_dim_affinities(solution, metric);

        // compute gradient
        std::vector<std::vector<double>> gradients =
            kullback_leibler_gradient(symmetrized, q_ij, solution, metric);

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
void clusterxx::TSNE<Metric>::fit(const std::vector<std::vector<double>> X) {
    __fit(X);
}

template <typename Metric>
std::vector<std::vector<double>> clusterxx::TSNE<Metric>::fit_transform(
    const std::vector<std::vector<double>> X) {
    __fit(X);
    return __features;
}

template <typename Metric>
std::pair<int, int> clusterxx::TSNE<Metric>::get_shape() {
    return __shape;
}

template <typename Metric>
std::vector<std::vector<double>> clusterxx::TSNE<Metric>::get_features() {
    return __features;
}
