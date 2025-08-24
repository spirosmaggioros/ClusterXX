#ifndef CLUSTERXX_METRICS_METRICS_HPP
#define CLUSTERXX_METRICS_METRICS_HPP

#include "clusterxx/metrics/utils.hpp"
#include <assert.h>
#include <math.h>
#include <vector>

namespace clusterxx {
namespace metrics {
struct euclidean_distance {
    double operator()(const std::vector<double> &X,
                      const std::vector<double> &Y) const {
        assert(!X.empty());
        assert(!Y.empty());
        assert(X.size() == Y.size());

        double dist = 0.0;
        for (size_t i = 0; i < X.size(); i++) {
            dist += (X[i] - Y[i]) * (X[i] - Y[i]);
        }

        return sqrt(dist);
    }
};

struct squared_euclidean_distance {
    double operator()(const std::vector<double> &X,
                      const std::vector<double> &Y) const {
        assert(!X.empty());
        assert(!Y.empty());
        assert(X.size() == Y.size());

        double dist = 0.0;
        for (size_t i = 0; i < X.size(); i++) {
            dist += (X[i] - Y[i]) * (X[i] - Y[i]);
        }

        return dist;
    }
};

struct chebyshev_distance {
    double operator()(const std::vector<double> &X,
                      const std::vector<double> &Y) const {
        assert(!X.empty());
        assert(!Y.empty());
        assert(X.size() == Y.size());

        double dist = 0.0;
        for (size_t i = 0; i < X.size(); i++) {
            dist = std::max(dist, abs(X[i] - Y[i]));
        }

        return dist;
    }
};

struct manhattan_distance {
    double operator()(const std::vector<double> &X,
                      const std::vector<double> &Y) const {
        assert(!X.empty());
        assert(!Y.empty());
        assert(X.size() == Y.size());

        double dist = 0.0;
        for (size_t i = 0; i < X.size(); i++) {
            dist += abs(X[i] - Y[i]);
        }

        return dist;
    }
};
} // namespace metrics

namespace pairwise_distances {
struct euclidean_distances {
    std::vector<std::vector<double>>
    operator()(const std::vector<std::vector<double>> &X,
               const std::vector<std::vector<double>> &Y = {}) const {
        return compute_pairwise_distances<
            clusterxx::metrics::euclidean_distance>(X, Y);
    }
};

struct manhattan_distances {
    std::vector<std::vector<double>>
    operator()(const std::vector<std::vector<double>> &X,
               const std::vector<std::vector<double>> &Y = {}) const {
        return compute_pairwise_distances<
            clusterxx::metrics::manhattan_distance>(X, Y);
    }
};

struct squared_euclidean_distances {
    std::vector<std::vector<double>>
    operator()(const std::vector<std::vector<double>> &X,
               const std::vector<std::vector<double>> &Y = {}) const {
        return compute_pairwise_distances<
            clusterxx::metrics::squared_euclidean_distance>(X, Y);
    }
};

struct chebyshev_distances {
    std::vector<std::vector<double>>
    operator ()(const std::vector<std::vector<double>> &X,
                const std::vector<std::vector<double>> &Y = {}) const {
        return compute_pairwise_distances<clusterxx::metrics::chebyshev_distance>(X, Y);
    }
};
} // namespace pairwise_distances
} // namespace clusterxx

#endif
