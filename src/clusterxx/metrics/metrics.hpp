#ifndef CLUSTERXX_METRICS_METRICS_HPP
#define CLUSTERXX_METRICS_METRICS_HPP

#include "clusterxx/metrics/utils.hpp"
#include <assert.h>
#include <math.h>
#include <vector>
#include <armadillo>

namespace clusterxx {
namespace metrics {
struct euclidean_distance {
    double operator()(const arma::vec &X,
                      const arma::vec &Y) const {
        assert(!X.empty());
        assert(!Y.empty());
        assert(X.n_rows == Y.n_rows);

        return arma::norm2est(X - Y, 2);
    }
};

struct squared_euclidean_distance {
    double operator()(const arma::vec &X,
                      const arma::vec &Y) const {
        assert(!X.empty());
        assert(!Y.empty());
        assert(X.n_rows == Y.n_rows);

        return arma::norm2est(X - Y, 1);
    }
};

struct chebyshev_distance {
    double operator()(const arma::vec &X,
                      const arma::vec &Y) const {
        assert(!X.empty());
        assert(!Y.empty());
        assert(X.n_rows == Y.n_rows);

        return arma::max(arma::abs(X - Y));
    }
};

struct manhattan_distance {
    double operator()(const arma::vec &X,
                      const arma::vec &Y) const {
        assert(!X.empty());
        assert(!Y.empty());
        assert(X.n_rows == Y.n_rows);

        return arma::sum(arma::abs(X - Y));
    }
};
} // namespace metrics

namespace pairwise_distances {
struct euclidean_distances {
    arma::mat
    operator()(const arma::mat &X,
               const arma::mat &Y) const {
        assert(!X.empty());
        if (Y.empty()) {
            Y = X;
        }

        arma::vec norm_x = arma::sum(arma::square(_X), 1);
        arma::vec norm_y = arma::sum(arma::square(_Y), 1);
        arma::mat dot_prod = _X * _Y.t();

        return arma::sqrt(arma::repmat(norm_x, 1, _Y.n_rows) +
                         arma::repmat(norm_y.t(), _X.n_rows, 1) -
                         2 * dot_prod);
    }
};

struct manhattan_distances {
    arma::mat
    operator()(const arma::mat &X,
               const arma::mat &Y) const {
        return arma::mat();
    }
};

struct squared_euclidean_distances {
    arma::mat
    operator()(const arma::mat &X,
               const arma::mat &Y) const {
        return arma::mat();
    }
};

struct chebyshev_distances {
    arma::mat
    operator ()(const arma::mat &X,
                const arma::mat &Y) const {
        return arma::mat();
    }
};
} // namespace pairwise_distances
} // namespace clusterxx

#endif
