#ifndef CLUSTERXX_METRICS_METRICS_HPP
#define CLUSTERXX_METRICS_METRICS_HPP

#include <armadillo>
#include <assert.h>
#include <float.h>
#include <math.h>

namespace clusterxx {
namespace metrics {
struct euclidean_distance {
    double operator()(const arma::vec &X, const arma::vec &Y) const {
        assert(!X.empty());
        assert(!Y.empty());
        assert(X.n_rows == Y.n_rows);

        return arma::norm(X - Y, 2);
    }

    double p() { return 2; }
};

struct squared_euclidean_distance {
    double operator()(const arma::vec &X, const arma::vec &Y) const {
        assert(!X.empty());
        assert(!Y.empty());
        assert(X.n_rows == Y.n_rows);

        return arma::norm(X - Y, 1);
    }
};

struct chebyshev_distance {
    double operator()(const arma::vec &X, const arma::vec &Y) const {
        assert(!X.empty());
        assert(!Y.empty());
        assert(X.n_rows == Y.n_rows);

        return arma::max(arma::abs(X - Y));
    }
};

struct manhattan_distance {
    double operator()(const arma::vec &X, const arma::vec &Y = {}) const {
        assert(!X.empty());
        assert(!Y.empty());
        assert(X.n_rows == Y.n_rows);

        return arma::sum(arma::abs(X - Y));
    }

    double p() { return 1; }
};
} // namespace metrics

namespace pairwise_distances {
struct euclidean_distances {
    arma::mat operator()(const arma::mat &X, const arma::mat &Y = {}) {
        assert(!X.empty());
        const arma::mat &_Y = Y.empty() ? X : Y;

        arma::vec norm_x = arma::sum(arma::square(X), 1);
        arma::vec norm_y = arma::sum(arma::square(_Y), 1);

        arma::mat dot_prod = X * _Y.t();

        arma::mat result = -2 * dot_prod;
        result.each_col() += norm_x;
        result.each_row() += norm_y.t();

        result.transform(
            [](double val) { return std::sqrt(std::max(0.0, val)); });

        return result;
    }
};

struct manhattan_distances {
    arma::mat operator()(const arma::mat &X, const arma::mat &Y = {}) const {
        assert(!X.empty());
        const arma::mat &_Y = Y.empty() ? X : Y;

        arma::mat distances(X.n_rows, _Y.n_rows);

        for (arma::uword i = 0; i < X.n_rows; i++) {
            arma::rowvec x_row = X.row(i);
            arma::mat diff = -_Y;
            diff.each_row() += x_row;
            diff = arma::abs(diff);
            distances.row(i) = arma::sum(diff, 1).t();
        }

        return distances;
    }
};

struct squared_euclidean_distances {
    arma::mat operator()(const arma::mat &X, const arma::mat &Y = {}) const {
        assert(!X.empty());
        const arma::mat &_Y = Y.empty() ? X : Y;

        arma::vec norm_x = arma::sum(arma::square(X), 1);
        arma::vec norm_y = arma::sum(arma::square(_Y), 1);
        arma::mat dot_prod = X * _Y.t();

        arma::mat result = -2 * dot_prod;
        result.each_col() += norm_x;
        result.each_row() += norm_y.t();

        return result;
    }
};

struct chebyshev_distances {
    arma::mat operator()(const arma::mat &X, const arma::mat &Y = {}) const {
        assert(!X.empty());
        const arma::mat &_Y = Y.empty() ? X : Y;

        arma::mat distances(X.n_rows, _Y.n_rows);

        for (arma::uword i = 0; i < X.n_rows; i++) {
            arma::rowvec x_row = X.row(i);
            arma::mat diff = -_Y;
            diff.each_row() += x_row;
            diff = arma::abs(diff);
            distances.row(i) = arma::max(diff, 1);
        }

        return distances;
    }
};
} // namespace pairwise_distances

namespace clustering {
template <typename Metric = clusterxx::pairwise_distances::euclidean_distances>
struct silhouette_samples {
    std::vector<double> operator()(const arma::mat &X,
                                   const std::vector<int> &labels) {
        Metric metric;
        std::unordered_map<int, int> cluster_size;
        for (const auto &label : labels) {
            cluster_size[label]++;
        }

        std::unordered_map<int, std::vector<double>> dists_point_to_cluster;
        arma::mat pairwise_dists = metric(X);
        for (size_t i = 0; i < X.n_rows; i++) {
            for (size_t j = 0; j < X.n_rows; j++) {
                if (dists_point_to_cluster.find(i) ==
                    dists_point_to_cluster.end()) {
                    dists_point_to_cluster[i].resize(cluster_size.size(), 0.0);
                }
                dists_point_to_cluster[i][labels[j]] += pairwise_dists(i, j);
            }
        }
        std::vector<double> A(X.n_rows), B(X.n_rows);
        for (size_t i = 0; i < X.n_rows; i++) {
            A[i] = (1.0 / (cluster_size[labels[i]] - 1)) *
                   dists_point_to_cluster[i][labels[i]];
            B[i] = DBL_MAX;
            for (size_t j = 0; j < cluster_size.size(); j++) {
                if (j == labels[i]) {
                    continue;
                }
                B[i] = std::min(B[i], (1.0 / cluster_size[j]) *
                                          dists_point_to_cluster[i][j]);
            }
        }

        std::vector<double> SC;
        for (size_t i = 0; i < X.n_rows; i++) {
            if (cluster_size[labels[i]] > 1) {
                SC.push_back((B[i] - A[i]) / std::max(A[i], B[i]));
            } else {
                SC.push_back(0.0);
            }
        }

        return SC;
    }
};

template <typename Metric = clusterxx::pairwise_distances::euclidean_distances>
struct silhouette_score {
    double operator()(const arma::mat &X, const std::vector<int> &labels) {
        clusterxx::clustering::silhouette_samples<Metric> silhouette_samples;
        std::vector<double> SC = silhouette_samples(X, labels);
        return std::accumulate(SC.begin(), SC.end(), 0.0) / X.n_rows;
    }
};
} // namespace clustering
} // namespace clusterxx

#endif
