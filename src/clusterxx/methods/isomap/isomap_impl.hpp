#ifndef CLUSTERXX_METHODS_ISOMAP_IMPL_HPP
#define CLUSTERXX_METHODS_ISOMAP_IMPL_HPP

#include "clusterxx/data_structures/graph/graph.hpp"
#include "isomap.hpp"

template <class NeighAlgorithm>
clusterxx::isomap<NeighAlgorithm>::isomap(const unsigned int &n_neighbors,
                                          const double &radius,
                                          const unsigned int &n_components,
                                          const std::string &path_method)
    : __n_neighbors(n_neighbors), __radius(radius),
      __n_components(n_components), __path_method(path_method) {
    assert(radius >= 0.0);
    if (radius > 0.0) {
        assert(n_neighbors == 0); // Please use either radius or n_neighbors
    } else if (radius == 0) {
        assert(n_neighbors > 0);
    }
}

template <class NeighAlgorithm>
void clusterxx::isomap<NeighAlgorithm>::__fit(const arma::mat &X) {
    assert(!X.empty());
    __neigh_algorithm = std::make_unique<NeighAlgorithm>(X);
    Graph d_g("undirected");

    for (size_t i = 0; i < X.n_rows; i++) {
        std::vector<int> _inds;
        std::vector<double> _dists;
        if (__radius > 0.0) {
            std::tie(_inds, _dists) =
                __neigh_algorithm->query_radius(X.row(i).t(), __radius);
        } else {
            std::tie(_inds, _dists) =
                __neigh_algorithm->query(X.row(i).t(), __n_neighbors);
        }
        for (size_t j = 0; j < _inds.size(); j++) {
            if (_inds[j] != i) {
                d_g.insert_edge(i, _inds[j], _dists[j]);
            }
        }
    }

    int64_t E = d_g.n_edges();
    int64_t N = d_g.n_nodes();
    arma::mat D(N, N);
    if ((__path_method == "auto" &&
         N * N * N >= N * E + N * N * std::log2(N)) ||
        __path_method == "D") {
        std::vector<std::vector<double>> t_d_g =
            d_g.dijkstra_all_shortest_paths();
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++) {
                D(i, j) = t_d_g[i][j];
            }
        }
    } else {
        std::vector<double> t_d_g = d_g.floyd_warshall_all_shortest_paths();
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++) {
                D(i, j) = t_d_g[N * i + j];
            }
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
    arma::uvec indices = arma::sort_index(eigval, "descend");
    indices = indices.head(__n_components);
    arma::mat Y(N, __n_components);

    for (int i = 0; i < __n_components; i++) {
        double _l = eigval(indices(i));
        arma::vec v = eigvec.col(indices(i));
        Y.col(i) = v * std::sqrt(_l);
    }

    __features = Y;
    __shape.first = Y.n_rows;
    __shape.second = Y.n_cols;
}

template <class NeighAlgorithm>
void clusterxx::isomap<NeighAlgorithm>::fit(const arma::mat &X) {
    __features.clear();
    __fit(X);
}

template <class NeighAlgorithm>
arma::mat clusterxx::isomap<NeighAlgorithm>::fit_transform(const arma::mat &X) {
    __features.clear();
    __fit(X);
    return __features;
}

template <class NeighAlgorithm>
std::pair<int, int> clusterxx::isomap<NeighAlgorithm>::get_shape() const {
    assert(!__features.empty());
    return __shape;
}

template <class NeighAlgorithm>
arma::mat clusterxx::isomap<NeighAlgorithm>::get_features() const {
    assert(!__features.empty());
    return __features;
}

#endif
