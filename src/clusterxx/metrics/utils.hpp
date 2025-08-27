#ifndef CLUSTERXX_METRICS_UTILS_HPP
#define CLUSTERXX_METRICS_UTILS_HPP

#include <armadillo>
#include <assert.h>
#include <math.h>
#include <algorithm>
#include <thread>
#include <vector>

#include <iostream>

template <typename Metric>
void __compute_partial(
        const arma::mat &X,
        const arma::mat &Y,
        arma::mat &dist_vec,
        const size_t &range_from, const size_t &range_to) {
    Metric metric;

    if (Y.empty()) {
        for (size_t i = range_from; i <= range_to; i++) {
            for (size_t j = i; j < X.n_rows; j++) {
                arma::vec xi = X.row(i).t();
                arma::vec yj = Y.row(j).t();
                double d = metric(xi, yj);
                dist_vec(i, j) = d;
                dist_vec(j, i) = d;
            }
        }
    } else {
        for (size_t i = range_from; i <= range_to; i++) {
            for (size_t j = 0; j < Y.n_rows; j++) {
                arma::vec xi = X.row(i).t();
                arma::vec yj = Y.row(j).t();
                dist_vec(i, j) = metric(xi, yj);
            }
        }
    }
}

template <typename Metric>
arma::mat __multithread_compute_pairwise(const arma::mat &X, const arma::mat &Y, const int n_workers) {
    arma::mat dist;
    if (Y.empty()) {
        dist.resize(X.n_rows, X.n_rows);
    } else {
        assert(X.n_cols == Y.n_cols);
        dist.resize(X.n_rows, Y.n_rows);
    }

    std::vector<std::thread> threads;
    int range_from = 0;
    int window = (X.n_rows + n_workers - 1) / n_workers;
    for (int i = 0; i < n_workers; i++) {
        int range_to = std::min(range_from + window - 1, int(X.n_rows - 1));
        if (range_from <= range_to) {
            threads.push_back(std::thread([&X, &Y, &range_from, &range_to, &dist]{
                        __compute_partial<Metric>(X, Y, dist, range_from, range_to);
            }));
        }
        range_from = range_to + 1;
    }

    for (auto &thread: threads) {
        thread.join();
    }

    return dist;
}

#endif
