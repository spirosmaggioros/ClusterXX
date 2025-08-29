#ifndef CLUSTERXX_METHODS_DBSCAN_IMPL_HPP
#define CLUSTERXX_METHODS_DBSCAN_IMPL_HPP

#include "dbscan.hpp"

template <typename Metric, class Algorithm>
void clusterxx::DBSCAN<Metric, Algorithm>::__fit(const arma::mat &X) {
    assert(!X.empty());
    __algorithm = std::make_unique<Algorithm>(X);
    int cluster_id = 0;
    for (size_t i = 0; i < X.n_rows; i++) {
        const arma::vec &_curr_point = X.row(i).t();
        if (__assignments.find(i) == __assignments.end()) {
            auto [inds, _] = __algorithm->query_radius(_curr_point, __eps);
            if (inds.size() < __min_samples) {
                __assignments[i] = -1;
                continue;
            }

            for (auto &ind : inds) {
                __assignments[i] = cluster_id;
            }

            while (!inds.empty()) {
                auto current = inds.back();
                if (inds.back() == i) {
                    inds.pop_back();
                    continue;
                } else {
                    inds.pop_back();
                }
                

                const arma::vec &curr = X.row(current).t();
                auto [res, _] = __algorithm->query_radius(curr, __eps);

                if (res.size() >= __min_samples) {
                    for (size_t r = 0; r < res.size(); r++) {
                        if (__assignments.find(res[r]) == __assignments.end()) {
                            inds.push_back(res[r]);
                            __assignments[res[r]] = cluster_id;
                        } else if (__assignments[res[r]] == -1) {
                            __assignments[res[r]] = cluster_id;
                        }
                    }
                }
            }
            cluster_id++;
        }
    }

    for (size_t i = 0; i < X.n_rows; i++) {
        __labels.push_back(__assignments[i]);
    }
}

template <typename Metric, class Algorithm>
void clusterxx::DBSCAN<Metric, Algorithm>::fit(const arma::mat &X) {
    __assignments.clear();
    __labels.clear();
    __fit(X);
}

template <typename Metric, class Algorithm>
std::vector<int>
clusterxx::DBSCAN<Metric, Algorithm>::fit_predict(const arma::mat &X) {
    __assignments.clear();
    __labels.clear();
    __fit(X);
    return __labels;
}

template <typename Metric, class Algorithm>
std::vector<int>
clusterxx::DBSCAN<Metric, Algorithm>::predict(const arma::mat &X) {
    std::cout << "[WARNING] predict() function is not implemented for this "
                 "class. Use fit_predict() instead"
              << '\n';
    // Not implemented
    return {};
}

template <typename Metric, class Algorithm>
std::vector<int> clusterxx::DBSCAN<Metric, Algorithm>::get_labels() {
    assert(!__assignments.empty());
    assert(!__labels.empty());
    return __labels;
}

#endif
