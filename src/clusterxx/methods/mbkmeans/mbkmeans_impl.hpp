#ifndef CLUSTERXX_METHODS_MBKMEANS_IMPL_HPP
#define CLUSTERXX_METHODS_MBKMEANS_IMPL_HPP

#include "mbkmeans.hpp"

template <typename Metric>
void clusterxx::MiniBatchKMeans<Metric>::__init_centroids() {
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
void clusterxx::MiniBatchKMeans<Metric>::__assign_labels(
    const arma::mat &X, const arma::uvec &batches) {
    arma::mat X_batches;
    if (!batches.empty()) {
        X_batches.resize(batches.n_rows, X.n_cols);
        int i = 0;
        for (const auto &ind : batches) {
            X_batches.row(i++) = X.row(ind);
        }
    } else {
        X_batches = X;
    }
    const arma::mat pairwise_dist = metric(X_batches, __centroids);
    for (size_t feat = 0; feat < pairwise_dist.n_rows; feat++) {
        auto selected_centroid = pairwise_dist.row(feat).index_min();
        int _feature = batches.empty() ? feat : batches(feat);
        __assignments[selected_centroid].push_back(_feature);
        __labels[_feature] = selected_centroid;
    }
}

template <typename Metric>
void clusterxx::MiniBatchKMeans<Metric>::__fit(const arma::mat &X) {
    assert(!X.empty());
    assert(__n_clusters <= X.n_rows);
    __center_counts.resize(__n_clusters);
    if (__batch_size > X.n_rows) {
        __batch_size = X.n_rows;
    }
    __features = X;
    __labels.resize(__features.n_rows);
    __init_centroids();

    for (int i = 0; i < __max_iter; i++) {
        __assignments.clear();
        arma::uvec batches = arma::randperm(X.n_rows, __batch_size);
        __assign_labels(X, batches);

        for (const auto &x : batches) {
            int center = __labels[x];
            __center_counts[center]++;
            double lr = 1.0 / __center_counts[center];
            for (size_t dim = 0; dim < __centroids.n_cols; dim++) {
                __centroids(center, dim) = (1 - lr) * __centroids(center, dim) +
                                           lr * __features(x, dim);
            }
        }
    }
}

template <typename Metric>
clusterxx::MiniBatchKMeans<Metric>::MiniBatchKMeans(
    const uint16_t n_clusters, const std::string init, const uint32_t max_iter,
    const uint32_t batch_size, std::optional<int> random_state)
    : __n_clusters(n_clusters), __init(init), __max_iter(max_iter),
      __batch_size(batch_size), __random_state(random_state) {
    assert(init == "k-means++" || init == "random");
}

template <typename Metric>
void clusterxx::MiniBatchKMeans<Metric>::fit(const arma::mat &X) {
    __assignments.clear();
    __centroids.clear();
    __labels.clear();
    __fit(X);
}

template <typename Metric>
std::vector<int>
clusterxx::MiniBatchKMeans<Metric>::fit_predict(const arma::mat &X) {
    __assignments.clear();
    __centroids.clear();
    __labels.clear();
    __fit(X);
    return __labels;
}

template <typename Metric>
std::vector<int>
clusterxx::MiniBatchKMeans<Metric>::predict(const arma::mat &X) {
    assert(!X.empty());
    assert(!__centroids.empty());
    assert(__centroids.n_cols == X.n_cols);

    __labels.clear();
    __labels.resize(X.size());
    __assign_labels(X, arma::uvec());
    return __labels;
}

template <typename Metric>
std::vector<int> clusterxx::MiniBatchKMeans<Metric>::get_labels() const {
    assert(!__labels.empty());
    assert(!__centroids.empty());
    return __labels;
}

template <typename Metric>
arma::mat clusterxx::MiniBatchKMeans<Metric>::get_centroids() const {
    assert(!__labels.empty());
    assert(!__centroids.empty());
    return __centroids;
}

#endif
