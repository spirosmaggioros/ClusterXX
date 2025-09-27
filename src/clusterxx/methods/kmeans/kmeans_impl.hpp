#ifndef CLUSTERXX_METHODS_KMEANS_IMPL_HPP
#define CLUSTERXX_METHODS_KMEANS_IMPL_HPP

#include "kmeans.hpp"
#include <algorithm>
#include <assert.h>
#include <ranges>

template <typename Metric>
clusterxx::KMeans<Metric>::KMeans(const uint16_t n_clusters,
                                  const uint32_t max_iter, std::string init,
                                  std::optional<int> random_state)
    : __n_clusters(n_clusters), __max_iter(max_iter), __init(init),
      __random_state(random_state) {
    assert(max_iter > 0);
    assert(n_clusters > 0);
    assert(init == "k-means++" || init == "random");
}

template <typename Metric> void clusterxx::KMeans<Metric>::__init_centroids() {
    if (__init == "random") {
        __centroids.resize(__n_clusters, __features.n_cols);
        int seed = __random_state.value_or(-1);
        arma::mat shuffled;
        if (seed == -1) {
            shuffled = arma::shuffle(__features);
        } else {
            arma::arma_rng::set_seed(seed);
            shuffled = arma::shuffle(__features);
        }
        for (int i = 0; i < __n_clusters; i++) {
            __centroids.row(i) = shuffled.row(i);
        }
    } else { // k-means++
        __centroids.resize(1, __features.n_cols);
        int _rand = rand() % (__features.n_rows - 1);
        int _curr_centroid_idx = 0;
        __centroids.row(_curr_centroid_idx++) = __features.row(_rand);
        clusterxx::pairwise_distances::squared_euclidean_distances
            squared_eucl_dist;
        while (__centroids.n_rows < __n_clusters) {
            arma::mat pairwise_dist =
                squared_eucl_dist(__features, __centroids);
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
            __centroids.row(_curr_centroid_idx++) = __features.row(selected);
        }
    }
}

template <typename Metric>
void clusterxx::KMeans<Metric>::__assign_labels(const arma::mat &X) {
    const arma::mat pairwise_dist = metric(X, __centroids);
    for (size_t feat = 0; feat < pairwise_dist.n_rows; feat++) {
        auto selected_centroid = pairwise_dist.row(feat).index_min();
        __assignments[selected_centroid].push_back(feat);
        __labels[feat] = selected_centroid;
    }
}

template <typename Metric>
arma::mat clusterxx::KMeans<Metric>::__recalc_centroids() {
    arma::mat new_centroids(__n_clusters, __features.n_cols);
    int _curr_centroid_idx = 0;
    for (const auto &[centroid, points] : __assignments) {
        arma::vec _curr_mean(__features.n_cols);
        for (const auto &point : points) {
            for (size_t i = 0; i < __features.n_cols; i++) {
                _curr_mean(i) += __features(point, i);
            }
        }

        for (auto &x : _curr_mean) {
            x /= int(points.size());
        }
        new_centroids.row(_curr_centroid_idx++) = _curr_mean.t();
    }

    return new_centroids;
}

template <typename Metric>
void clusterxx::KMeans<Metric>::__fit(const arma::mat &X) {
    assert(!X.empty());
    assert(__n_clusters <= X.n_rows);

    __features = X;
    __labels.resize(__features.n_rows);
    __init_centroids();

    for (int i = 0; i < __max_iter; i++) {
        __assignments.clear();
        __assign_labels(__features);

        arma::mat new_centroids = __recalc_centroids();
        if (arma::approx_equal(__centroids, new_centroids, "absdiff", 1e-5)) {
            break;
        }

        __centroids = new_centroids;
    }
}

template <typename Metric>
void clusterxx::KMeans<Metric>::fit(const arma::mat &X) {
    __assignments.clear();
    __centroids.clear();
    __labels.clear();
    __fit(X);
}

template <typename Metric>
std::vector<int> clusterxx::KMeans<Metric>::fit_predict(const arma::mat &X) {
    __assignments.clear();
    __centroids.clear();
    __labels.clear();
    __fit(X);
    return __labels;
}

template <typename Metric>
std::vector<int> clusterxx::KMeans<Metric>::predict(const arma::mat &X) {
    assert(!X.empty());
    assert(!__centroids.empty());
    assert(__centroids.n_cols == X.n_cols);

    __labels.clear();
    __labels.resize(X.size());
    __assign_labels(X);
    return __labels;
}

template <typename Metric>
std::vector<int> clusterxx::KMeans<Metric>::get_labels() const {
    assert(!__labels.empty());
    assert(!__centroids.empty());
    return __labels;
}

template <typename Metric>
arma::mat clusterxx::KMeans<Metric>::get_centroids() const {
    assert(!__labels.empty());
    assert(!__centroids.empty());
    return __centroids;
}

#endif
