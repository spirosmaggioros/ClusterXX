#ifndef CLUSTERXX_METHODS_DBSCAN_IMPL_HPP
#define CLUSTERXX_METHODS_DBSCAN_IMPL_HPP

#include "dbscan.hpp"

template <class Algorithm>
void clusterxx::DBSCAN<Algorithm>::__fit(const arma::mat &X) {
    assert(!X.empty());
    __algorithm = std::make_unique<Algorithm>(X, __leaf_size);
    int cluster_id = 0;
    for (size_t i = 0; i < X.n_rows; i++) {
        if (__assignments.find(i) == __assignments.end()) {
            const arma::vec &_curr_point = X.row(i).t();
            auto [inds, _] = __algorithm->query_radius(_curr_point, __eps);
            if (inds.size() < __min_samples) {
                __assignments[i] = -1;
                continue;
            }

            for (const auto &ind : inds) {
                __assignments[ind] = cluster_id;
            }

            while (!inds.empty()) {
                if (inds.back() == i) {
                    inds.pop_back();
                    continue;
                }

                const arma::vec &curr = X.row(inds.back()).t();
                auto [res, _] = __algorithm->query_radius(curr, __eps);

                if (res.size() >= __min_samples) {
                    for (const auto &n : res) {
                        if (__assignments.find(n) == __assignments.end()) {
                            inds.push_back(n);
                            __assignments[n] = cluster_id;
                        } else if (__assignments[n] == -1) {
                            __assignments[n] = cluster_id;
                        }
                    }
                }
                inds.pop_back();
            }
            cluster_id++;
        }
    }

    for (size_t i = 0; i < X.n_rows; i++) {
        if (__assignments.find(i) == __assignments.end()) {
            __labels.push_back(-1);
        } else {
            __labels.push_back(__assignments[i]);
        }
    }
}

template <class Algorithm>
void clusterxx::DBSCAN<Algorithm>::fit(const arma::mat &X) {
    __assignments.clear();
    __labels.clear();
    __fit(X);
}

template <class Algorithm>
std::vector<int>
clusterxx::DBSCAN<Algorithm>::fit_predict(const arma::mat &X) {
    __assignments.clear();
    __labels.clear();
    __fit(X);
    return __labels;
}

template <class Algorithm>
std::vector<int>
clusterxx::DBSCAN<Algorithm>::predict(const arma::mat &X) {
    std::cout << "[WARNING] predict() function is not implemented for this "
                 "class. Use fit_predict() instead"
              << '\n';
    // Not implemented
    return {};
}

template <class Algorithm>
std::vector<int> clusterxx::DBSCAN<Algorithm>::get_labels() const {
    assert(!__assignments.empty());
    assert(!__labels.empty());
    return __labels;
}

#endif
