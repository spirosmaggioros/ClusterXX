#ifndef CLUSTERXX_METRICS_UTILS_HPP
#define CLUSTERXX_METRICS_UTILS_HPP

#include <assert.h>
#include <math.h>
#include <vector>

template <typename Metric>
std::vector<std::vector<double>>
compute_pairwise_distances(const std::vector<std::vector<double>> &X,
                           const std::vector<std::vector<double>> Y = {},
                           const bool squared = true) {
    Metric metric;

    if (Y.empty()) { // only compute pairwise distances for X
        std::vector<std::vector<double>> dist(
            X.size(), std::vector<double>(X.size(), 0.0));

        for (size_t i = 0; i < X.size(); i++) {
            for (size_t j = 0; j < X.size(); j++) {
                dist[i][j] = metric(X[i], X[j]);
            }
        }

        return dist;
    } else {
        assert(X[0].size() == Y[0].size());
        std::vector<std::vector<double>> dist(X.size(),
                                              std::vector<double>(Y.size()));
        for (size_t i = 0; i < X.size(); i++) {
            for (size_t j = 0; j < Y.size(); j++) {
                dist[i][j] = metric(X[i], Y[j]);
            }
        }

        return dist;
    }
}

#endif
