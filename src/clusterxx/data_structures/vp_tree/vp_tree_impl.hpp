#ifndef CLUSTERXX_DATA_STRUCTURES_VP_TREE_IMPL_HPP
#define CLUSTERXX_DATA_STRUCTURES_VP_TREE_IMPL_HPP

#include "vp_tree.hpp"

template <typename Metric>
std::unique_ptr<typename clusterxx::vp_tree<Metric>::vp_node>
clusterxx::vp_tree<Metric>::__initialize(int64_t lower, int64_t upper) {
    if (upper == lower) {
        return nullptr;
    }

    std::unique_ptr<vp_node> nn = std::make_unique<vp_node>();
    nn->__index = lower;

    if (upper - lower > 1) {
        size_t _randn = rand() % (upper - lower - 1) + lower;
        __in_features.swap_rows(lower, _randn);
        std::vector<size_t> _indices(upper - lower);
        std::iota(_indices.begin(), _indices.end(), lower);
        size_t median_offset = (_indices.size()) / 2;
        size_t median_index = _indices[median_offset];

        std::nth_element(_indices.begin(), _indices.begin() + median_offset,
                         _indices.end(), [&](size_t a, size_t b) {
                             return metric(__in_features.row(lower).t(),
                                           __in_features.row(a).t()) <
                                    metric(__in_features.row(lower).t(),
                                           __in_features.row(b).t());
                         });

        nn->__mu = metric(__in_features.row(lower).t(),
                          __in_features.row(median_index).t());

        nn->left = __initialize(lower + 1, median_index);
        nn->right = __initialize(median_index, upper);
    }

    return nn;
}

template <typename Metric>
void clusterxx::vp_tree<Metric>::__k_nearest_neighbors(
    std::unique_ptr<vp_node> &node, const arma::vec &x, MaxHeap &heap,
    const uint32_t &k, double &tau) {
    if (node == nullptr) {
        return;
    }
    double _dist = metric(__in_features.row(node->__index).t(), x);

    if (_dist < tau) {
        if (heap.size() == k) {
            heap.pop();
        }
        heap.push({_dist, node->__index});
        if (heap.size() == k) {
            tau = heap.top().first;
        }
    }

    if (node->left == nullptr && node->right == nullptr) {
        return;
    }

    if (_dist < node->__mu) {
        if (_dist - tau <= node->__mu) {
            __k_nearest_neighbors(node->left, x, heap, k, tau);
        }
        if (_dist + tau >= node->__mu) {
            __k_nearest_neighbors(node->right, x, heap, k, tau);
        }
    } else {
        if (_dist + tau >= node->__mu) {
            __k_nearest_neighbors(node->right, x, heap, k, tau);
        }
        if (_dist - tau <= node->__mu) {
            __k_nearest_neighbors(node->left, x, heap, k, tau);
        }
    }
}

template <typename Metric>
uint64_t clusterxx::vp_tree<Metric>::__depth(std::unique_ptr<vp_node> &root) {
    if (!root) {
        return 0;
    }

    return 1 + std::max(__depth(root->left), __depth(root->right));
}

template <typename Metric>
clusterxx::vp_tree<Metric>::vp_tree(const arma::mat &X) : __in_features(X) {
    assert(!X.empty());
    assert(metric.p() > 0);
    __root = __initialize(0, X.n_rows);
}

template <typename Metric>
std::pair<std::vector<int>, std::vector<double>>
clusterxx::vp_tree<Metric>::query(const arma::vec &X, const uint32_t &k) {
    MaxHeap heap;
    double _tau = DBL_MAX;
    __k_nearest_neighbors(__root, X, heap, k, _tau);

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

template <typename Metric> uint64_t clusterxx::vp_tree<Metric>::depth() {
    return __depth(__root);
}

#endif
