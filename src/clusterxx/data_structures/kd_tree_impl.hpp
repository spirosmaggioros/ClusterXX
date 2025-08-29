#ifndef CLUSTERXX_DATA_STRUCTURES_KD_TREE_IMPL_HPP
#define CLUSTERXX_DATA_STRUCTURES_KD_TREE_IMPL_HPP

#include "kd_tree.hpp"

template <typename Metric>
std::unique_ptr<typename clusterxx::kd_tree<Metric>::kd_node> clusterxx::kd_tree<Metric>::__initialize(
        const arma::mat &X, std::vector<size_t> &indices, int depth) {
    if (indices.empty()) {
        return nullptr;
    }
    int axis = depth % X.n_cols;

    std::vector<size_t> _idx_copy = indices;
    size_t mid = indices.size() / 2;
    std::nth_element(_idx_copy.begin(), _idx_copy.begin() + mid, _idx_copy.end(),
            [&](size_t a, size_t b){
            return X.row(a)(axis) < X.row(b)(axis);
    });

    size_t median = _idx_copy[mid];
    auto _nn = std::make_unique<kd_node>(X.row(median).t(), median);

    std::vector<size_t> _left(_idx_copy.begin(), _idx_copy.begin() + mid);
    std::vector<size_t> _right(_idx_copy.begin() + mid + 1, _idx_copy.end());

    _nn->left = __initialize(X, _left, depth + 1);
    _nn->right = __initialize(X, _right, depth + 1);

    return _nn;
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
    double bound = 0.0;
    if (metric.p() == 0 || metric.p() == 1) {
        bound = std::abs(diff);
    } else if (metric.p() == 2) {
        bound = diff * diff;
    }
    if (heap.size() < k || bound < heap.top().first) {
        __k_nearest_neighbors(_to_check, X, heap, depth + 1, k);
    }
}

template <typename Metric>
void clusterxx::kd_tree<Metric>::__radius_nearest_neighbors(
    std::unique_ptr<kd_node> &node, const arma::vec &X, 
    std::vector<double> &dists, std::vector<int> &inds,
    const double radius, const int depth) {
    if (!node) {
        return;
    }
    assert(X.n_cols == node->__feature_size);

    int axis = depth % X.n_cols;
    double dist = metric(node->__point, X);
    if (dist <= radius) {
        dists.push_back(dist);
        inds.push_back(node->__ind);
    }

    std::unique_ptr<kd_node> &_next_node =
        (X(axis) < node->__point(axis)) ? node->left : node->right;
    std::unique_ptr<kd_node> &_to_check =
        (X(axis) < node->__point(axis)) ? node->right : node->left;

    __radius_nearest_neighbors(_next_node, X, dists, inds, radius, depth + 1);

    double diff = X(axis) - node->__point(axis);
    double bound = 0.0;
    if (metric.p() == 0 || metric.p() == 1) {
        bound = std::abs(diff);
    } else if (metric.p() == 2) {
        bound = diff * diff;
    }

    if (bound <= radius) {
        __radius_nearest_neighbors(_to_check, X, dists, inds, radius, depth + 1);
    }
}

template <typename Metric>
std::pair<std::vector<int>, std::vector<double>>
clusterxx::kd_tree<Metric>::query(const arma::vec &X, const int &k) {
    MaxHeap heap;
    __k_nearest_neighbors(__root, X, heap, 0, k);
    std::vector<double> dists;
    std::vector<int> inds;

    while (!heap.empty()) {
        auto [dist, ind] = heap.top();
        dists.push_back(dist);
        inds.push_back(ind);
        heap.pop();
    }

    return std::make_pair(inds, dists);
}

template <typename Metric>
std::pair<std::vector<int>, std::vector<double>>
clusterxx::kd_tree<Metric>::query_radius(const arma::vec &X, const double &r) {
    std::vector<double> dists;
    std::vector<int> inds;
    __radius_nearest_neighbors(__root, X, dists, inds, r);

    return std::make_pair(inds, dists);
}

#endif
