#ifndef CLUSTERXX_METRICS_METRICS_HPP
#define CLUSTERXX_METRICS_METRICS_HPP
#define ARMA_USE_BLAS

#include <armadillo>
#include <assert.h>
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

    double p() { return 0; }
};

struct manhattan_distance {
    double operator()(const arma::vec &X, const arma::vec &Y) const {
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
    arma::mat operator()(const arma::mat &X, const arma::mat &Y) {
        assert(!X.empty());

        arma::mat _X = X;
        arma::mat _Y;
        if (Y.empty()) {
            _Y = _X;
        } else {
            _Y = Y;
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
    arma::mat operator()(const arma::mat &X, const arma::mat &Y) const {
        assert(!X.empty());

        arma::mat _X = X;
        arma::mat _Y;

        if (Y.empty()) {
            _Y = _X;
        } else {
            _Y = Y;
        }

        arma::mat distances(_X.n_rows, _Y.n_rows);

        for (arma::uword i = 0; i < _X.n_rows; i++) {
            arma::mat diff =
                arma::abs(arma::repmat(_X.row(i), _Y.n_rows, 1) - _Y);
            distances.row(i) = arma::sum(diff, 1).t();
        }

        return distances;
    }
};

struct squared_euclidean_distances {
    arma::mat operator()(const arma::mat &X, const arma::mat &Y) const {
        assert(!X.empty());

        arma::mat _X = X;
        arma::mat _Y;
        if (Y.empty()) {
            _Y = _X;
        } else {
            _Y = Y;
        }

        arma::vec norm_x = arma::sum(arma::square(_X), 1);
        arma::vec norm_y = arma::sum(arma::square(_Y), 1);
        arma::mat dot_prod = _X * _Y.t();

        return arma::repmat(norm_x, 1, Y.n_rows) +
               arma::repmat(norm_y.t(), X.n_rows, 1) - 2 * dot_prod;
    }
};

struct chebyshev_distances {
    arma::mat operator()(const arma::mat &X, const arma::mat &Y) const {
        assert(!X.empty());

        arma::mat _X = X;
        arma::mat _Y;
        if (Y.empty()) {
            _Y = _X;
        } else {
            _Y = Y;
        }

        arma::mat distances(_X.n_rows, _Y.n_rows);

        for (arma::uword i = 0; i < _X.n_rows; i++) {
            arma::mat diff =
                arma::abs(arma::repmat(_X.row(i), _Y.n_rows, 1) - _Y);
            distances.row(i) = arma::max(diff, 1);
        }

        return distances;
    }
};
} // namespace pairwise_distances
} // namespace clusterxx

#endif
