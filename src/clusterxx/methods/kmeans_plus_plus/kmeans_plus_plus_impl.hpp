#ifndef CLUSTERXX_METHODS_KMEANS_PLUS_PLUS_IMPL_HPP
#define CLUSTERXX_METHODS_KMEANS_PLUS_PLUS_IMPL_HPP

#include "clusterxx/metrics/metrics.hpp"
#include "kmeans_plus_plus.hpp"
#include <assert.h>

void clusterxx::kmeans_plus_plus::__fit(const arma::mat &X) {
    __centroids.resize(1, X.n_cols);
    int _rand = rand() % (X.n_rows - 1);
    int _curr_centroid_idx = 0;
    __centroids.row(_curr_centroid_idx++) = X.row(_rand);
    clusterxx::pairwise_distances::squared_euclidean_distances
        squared_eucl_dist;
    while (__centroids.n_rows < __n_clusters) {
        arma::mat pairwise_dist = squared_eucl_dist(X, __centroids);
        std::vector<double> probs;
        double probs_sum = 0.0;
        for (size_t i = 0; i < pairwise_dist.n_rows; i++) {
            auto min_element = pairwise_dist.row(i).min();
            probs.push_back(min_element);
            probs_sum += min_element;
        }

        auto dist_compare = [&probs_sum](double a, double b) {
            return (a / probs_sum) < (b / probs_sum);
        };
        auto max_prob = std::ranges::max_element(probs, dist_compare);
        int selected = std::ranges::distance(probs.begin(), max_prob);
        __centroids.insert_rows(__centroids.n_rows - 1, 1);
        __centroids.row(_curr_centroid_idx++) = X.row(selected);
    }
}

clusterxx::kmeans_plus_plus::kmeans_plus_plus(const uint32_t n_clusters)
    : __n_clusters(n_clusters) {
    assert(__n_clusters > 0);
}

void clusterxx::kmeans_plus_plus::fit(const arma::mat &X) {
    assert(!X.empty());
    __centroids.clear();
    __fit(X);
}

arma::mat clusterxx::kmeans_plus_plus::get_centroids() const {
    assert(!__centroids.empty());
    return __centroids;
}

#endif
