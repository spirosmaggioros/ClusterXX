#ifndef CLUSTERXX_METHODS_ISOMAP_IMPL_HPP
#define CLUSTERXX_METHODS_ISOMAP_IMPL_HPP

#include "clusterxx/data_structures/graph/graph.hpp"
#include "isomap.hpp"

template <typename Metric, class NeighAlgorithm>
clusterxx::isomap<Metric, NeighAlgorithm>::isomap(
    const unsigned int &n_neighbors, const double &radius,
    const unsigned int &n_components)
    : __n_neighbors(n_neighbors), __radius(radius),
      __n_components(n_components) {
    assert(radius >= 0.0);
    if (radius > 0.0) {
        assert(n_neighbors == 0); // Please use either radius or n_neighbors
    } else if (radius == 0) {
        assert(n_neighbors > 0);
    }
}

template <typename Metric, class NeighAlgorithm>
void clusterxx::isomap<Metric, NeighAlgorithm>::fit(const arma::mat &X) {
    assert(!X.empty());
    __neigh_algorithm = std::make_unique<NeighAlgorithm>(X);
    Graph d_g("undirected");

    for (size_t i = 0; i < X.n_rows; i++) {
        std::cout << X.row(i).t() << '\n';
        std::vector<int> _inds;
        std::vector<double> _dists;
        if (__radius > 0.0) {
            _inds, _dists =
                       __neigh_algorithm->query_radius(X.row(i).t(), __radius);
        } else {
            _inds,
                _dists = __neigh_algorithm->query(X.row(i).t(), __n_neighbors);
        }
        for (size_t j = 0; j < _inds.size(); j++) {
            if (_inds[j] != i) {
                d_g.insert_edge(i, j, _dists[j]);
            }
        }
    }

    std::vector<std::vector<double>> t_d_g = d_g.floyd_warshall();
    size_t N = t_d_g.size();
    arma::mat D(N, N);
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            D(i, j) = t_d_g[i][j];
        }
    }

    arma::mat S = arma::pow(D, 2);
    arma::mat I = arma::eye(N, N);
    arma::mat ones = arma::ones(N, N) / N;
    arma::mat H = I - ones;
    arma::mat t_d = -0.5 * H * S * H;

    arma::vec eigval;
    arma::mat eigvec;

    arma::eig_sym(eigval, eigvec, t_d);

    arma::mat Y(N, __n_components);

    std::cout << eigval << '\n';
    std::cout << " ######### " << '\n';
    std::cout << eigvec << '\n';
}

#endif
