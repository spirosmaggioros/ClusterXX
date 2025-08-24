#ifndef CLUSTERXX_METHODS_KMEANS_IMPL_HPP
#define CLUSTERXX_METHODS_KMEANS_IMPL_HPP

#include "kmeans.hpp"
#include <algorithm>
#include <assert.h>
#include <float.h>
#include <random>

template <typename Metric>
void clusterxx::KMeans<Metric>::__init_centroids(
    std::vector<std::vector<double>> features) {
    int seed = __random_state.value_or(-1);
    if (seed == -1) {
        std::random_device r;
        std::default_random_engine el(r());
        std::shuffle(features.begin(), features.end(), el);
    } else {
        std::default_random_engine el(seed);
        std::shuffle(features.begin(), features.end(), el);
    }

    for (int i = 0; i < __n_clusters; i++) {
        __centroids.push_back(features[i]);
    }
}

template <typename Metric>
void clusterxx::KMeans<Metric>::__assign_labels(
    const std::vector<std::vector<double>> &X) {
    for (size_t feat = 0; feat < X.size(); feat++) {
        double _min_dist = DBL_MAX;
        int _selected_centroid = 0;
        for (int cent = 0; cent < __n_clusters; cent++) {
            double dist = metric(X[feat], __centroids[cent]);
            if (dist < _min_dist) {
                _min_dist = dist;
                _selected_centroid = cent;
            }
        }
        __assignments[_selected_centroid].push_back(feat);
        __labels[feat] = _selected_centroid;
    }
}

template <typename Metric>
std::vector<std::vector<double>>
clusterxx::KMeans<Metric>::__recalc_centroids() {
    std::vector<std::vector<double>> new_centroids;
    for (const auto &[centroid, points] : __assignments) {
        std::vector<double> _curr_mean(__features[0].size(), 0.0);
        for (const auto &point : points) {
            for (size_t i = 0; i < __features[0].size(); i++) {
                _curr_mean[i] += __features[point][i];
            }
        }

        for (auto &x : _curr_mean) {
            x /= int(points.size());
        }
        new_centroids.push_back(_curr_mean);
    }

    return new_centroids;
}

template <typename Metric>
void clusterxx::KMeans<Metric>::__fit(
    const std::vector<std::vector<double>> &X) {
    assert(!X.empty());
    assert(__n_clusters <= X.size());

    __features = X;
    __labels.resize(__features.size());
    __init_centroids(X);

    for (int i = 0; i < __max_iter; i++) {
        __assignments.clear();
        __assign_labels(__features);

        std::vector<std::vector<double>> new_centroids = __recalc_centroids();
        if (new_centroids == __centroids) {
            break;
        }

        __centroids = new_centroids;
    }
}

template <typename Metric>
void clusterxx::KMeans<Metric>::fit(const std::vector<std::vector<double>> &X) {
    __fit(X);
}

template <typename Metric>
std::vector<int> clusterxx::KMeans<Metric>::fit_predict(
    const std::vector<std::vector<double>> &X) {
    __fit(X);
    return __labels;
}

template <typename Metric>
std::vector<int>
clusterxx::KMeans<Metric>::predict(const std::vector<std::vector<double>> &X) {
    assert(!X.empty());
    assert(!__centroids.empty());
    assert(__centroids[0].size() == X[0].size());

    __labels.clear();
    __labels.resize(X.size());
    __assign_labels(X);
    return __labels;
}

template <typename Metric>
std::vector<int> clusterxx::KMeans<Metric>::get_labels() {
    assert(!__labels.empty());
    assert(!__centroids.empty());
    return __labels;
}

template <typename Metric>
std::vector<std::vector<double>> clusterxx::KMeans<Metric>::get_centroids() {
    assert(!__labels.empty());
    assert(!__centroids.empty());
    return __centroids;
}

#endif
