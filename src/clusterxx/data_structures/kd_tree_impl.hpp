#ifndef CLUSTERXX_DATA_STRUCTURES_KD_TREE_IMPL_HPP
#define CLUSTERXX_DATA_STRUCTURES_KD_TREE_IMPL_HPP

#include "kd_tree.hpp"

template <typename Metric> void clusterxx::kd_tree<Metric>::__initialize() {
    __root = std::move(std::make_unique<kd_node>(__features.row(0).t(), 0));

    for (size_t i = 1; i < __features.n_rows; i++) {
        std::unique_ptr<kd_node> _nn =
            std::make_unique<kd_node>(__features.row(i).t(), i);
        __root->add(std::move(_nn));
    }
}

template <typename Metric>
void clusterxx::kd_tree<Metric>::add(const arma::vec &feature) {
    std::unique_ptr<kd_node> _nn = std::make_unique<kd_node>(feature.t());
    if (!__root) {
        __root = std::move(_nn);
    } else {
        __root->add(std::move(_nn));
    }
}

template <typename Metric>
void clusterxx::kd_tree<Metric>::__k_nearest_neighbors(
    std::unique_ptr<kd_node> &node, const arma::vec &X, MaxHeap &heap,
    const int depth, const int k) {
    if (!node) {
        return;
    }
    assert(X.n_cols == node->__feature_size);

    int axis = depth % X.n_cols;
    double dist = metric(node->__point, X);
    if (heap.size() < k) {
        heap.emplace(dist, node->__ind);
    } else if (dist < heap.top().first) {
        heap.pop();
        heap.emplace(dist, node->__ind);
    }

    std::unique_ptr<kd_node> &_next_node =
        (X(axis) < node->__point(axis)) ? node->left : node->right;
    std::unique_ptr<kd_node> &_to_check =
        (X(axis) < node->__point(axis)) ? node->right : node->left;

    __k_nearest_neighbors(_next_node, X, heap, depth + 1, k);

    double diff = X(axis) - node->__point(axis);
    double bound;
    if (metric.p() == 0) {
        bound = std::abs(diff);
    } else {
        bound = std::pow(std::abs(diff), metric.p());
    }
    if (heap.size() < k || bound < heap.top().first) {
        __k_nearest_neighbors(_to_check, X, heap, depth + 1, k);
    }
}

template <typename Metric>
void clusterxx::kd_tree<Metric>::__radius_nearest_neighbors(
    std::unique_ptr<kd_node> &node, const arma::vec &X, MaxHeap &heap,
    const double radius, const int depth) {
    if (!node) {
        return;
    }
    assert(X.n_cols == node->__feature_size);

    int axis = depth % X.n_cols;
    double dist = metric(node->__point, X);
    if (dist <= radius) {
        heap.emplace(dist, node->__ind);
    }

    std::unique_ptr<kd_node> &_next_node =
        (X(axis) < node->__point(axis)) ? node->left : node->right;
    std::unique_ptr<kd_node> &_to_check =
        (X(axis) < node->__point(axis)) ? node->right : node->left;

    __radius_nearest_neighbors(_next_node, X, heap, radius, depth + 1);

    double diff = X(axis) - node->__point(axis);
    double bound;
    if (metric.p() == 0) { // chebyshev distance
        bound = std::abs(diff);
    } else { // euclidean or manhattan
        bound = std::pow(std::abs(diff), metric.p());
    }

    if (bound <= radius) {
        __radius_nearest_neighbors(_to_check, X, heap, radius, depth + 1);
    }
}

template <typename Metric>
std::pair<std::vector<int>, std::vector<double>>
clusterxx::kd_tree<Metric>::query(const arma::vec &X, const int &k) {
    MaxHeap heap;
    __k_nearest_neighbors(__root, X, heap, 0, k);

    std::vector<std::pair<double, int>> _res;
    while (!heap.empty()) {
        _res.push_back(heap.top());
        heap.pop();
    }

    std::sort(_res.begin(), _res.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    std::vector<double> dists;
    std::vector<int> inds;
    for (auto &[dist, ind] : _res) {
        dists.push_back(dist);
        inds.push_back(ind);
    }

    return std::make_pair(inds, dists);
}

template <typename Metric>
std::pair<std::vector<int>, std::vector<double>>
clusterxx::kd_tree<Metric>::query_radius(const arma::vec &X, const double &r) {
    MaxHeap heap;
    __radius_nearest_neighbors(__root, X, heap, r);

    std::vector<std::pair<double, int>> _res;
    while (!heap.empty()) {
        _res.push_back(heap.top());
        heap.pop();
    }
    std::sort(_res.begin(), _res.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    std::vector<double> dists;
    std::vector<int> inds;
    for (auto &[dist, ind] : _res) {
        dists.push_back(dist);
        inds.push_back(ind);
    }

    return std::make_pair(inds, dists);
}

#endif
