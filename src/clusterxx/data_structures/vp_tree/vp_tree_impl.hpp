#ifndef CLUSTERXX_DATA_STRUCTURES_VP_TREE_IMPL_HPP
#define CLUSTERXX_DATA_STRUCTURES_VP_TREE_IMPL_HPP

#include "vp_tree.hpp"

template <typename Metric>
std::unique_ptr<typename clusterxx::vp_tree<Metric>::vp_node>
clusterxx::vp_tree<Metric>::__initialize(std::vector<size_t> &indices) {
    if (indices.empty()) {
        return nullptr;
    }

    std::unique_ptr<vp_node> _nn = std::make_unique<vp_node>();

    if (indices.size() > 1) {
        size_t randn = rand() % indices.size();
        std::swap(indices[0], indices[randn]);
    }
    _nn->__index = indices[0];

    if (indices.size() == 1) {
        return _nn;
    }

    std::vector<size_t> _idx_copy(indices.begin() + 1, indices.end());
    size_t mid = _idx_copy.size() / 2;
    std::nth_element(_idx_copy.begin(), _idx_copy.begin() + mid,
                     _idx_copy.end(), [&](size_t a, size_t b) {
                         return metric(__in_features.row(indices[0]).t(),
                                       __in_features.row(a).t()) <
                                metric(__in_features.row(indices[0]).t(),
                                       __in_features.row(b).t());
                     });
    _nn->__mu = metric(__in_features.row(indices[0]).t(),
                       __in_features.row(_idx_copy[mid]).t());
    std::vector<size_t> _left(_idx_copy.begin(), _idx_copy.begin() + mid);
    if (_idx_copy.size() >= mid) {
        std::vector<size_t> _right(_idx_copy.begin() + mid, _idx_copy.end());
        _nn->right = __initialize(_right);
    }

    _nn->left = __initialize(_left);

    return _nn;
}

template <typename Metric>
void clusterxx::vp_tree<Metric>::__k_nearest_neighbors(
    std::unique_ptr<vp_node> &node, const arma::vec &x, MaxHeap &heap,
    const uint32_t &k, double &tau) {
    if (node == nullptr) {
        return;
    }
    double _dist = metric(__in_features.row(node->__index).t(), x);

    if (heap.size() < k) {
        heap.push({_dist, node->__index});
        if (heap.size() == k) {
            tau = heap.top().first;
        }
    } else if (_dist < tau) {
        heap.pop();
        heap.push({_dist, node->__index});
        tau = heap.top().first;
    }

    if (node->left == nullptr && node->right == nullptr) {
        return;
    }

    if (_dist + tau < node->__mu) {
        __k_nearest_neighbors(node->left, x, heap, k, tau);
    }
    if (_dist - tau > node->__mu) {
        __k_nearest_neighbors(node->right, x, heap, k, tau);
    }
    if (node->__mu - tau <= _dist && _dist <= node->__mu + tau) {
        __k_nearest_neighbors(node->right, x, heap, k, tau);
        __k_nearest_neighbors(node->left, x, heap, k, tau);
    }
}

template <typename Metric>
void clusterxx::vp_tree<Metric>::__radius_nearest_neighbors(
    std::unique_ptr<vp_node> &node, const arma::vec &x,
    std::vector<double> &dists, std::vector<int> &inds, const double &radius) {
    if (node == nullptr) {
        return;
    }
    double _dist = metric(__in_features.row(node->__index).t(), x);

    if (_dist < radius) {
        dists.push_back(_dist);
        inds.push_back(node->__index);
    }

    if (node->left == nullptr && node->right == nullptr) {
        return;
    }

    if (_dist + radius < node->__mu) {
        __radius_nearest_neighbors(node->left, x, dists, inds, radius);
    }
    if (_dist - radius > node->__mu) {
        __radius_nearest_neighbors(node->right, x, dists, inds, radius);
    }
    if (node->__mu - radius <= _dist && _dist <= node->__mu + radius) {
        __radius_nearest_neighbors(node->right, x, dists, inds, radius);
        __radius_nearest_neighbors(node->left, x, dists, inds, radius);
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
    std::vector<size_t> indices(X.n_rows);
    std::iota(indices.begin(), indices.end(), 0);
    __root = __initialize(indices);
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

template <typename Metric>
std::pair<std::vector<int>, std::vector<double>>
clusterxx::vp_tree<Metric>::query_radius(const arma::vec &X, const double &r) {
    std::vector<double> dists;
    std::vector<int> inds;
    __radius_nearest_neighbors(__root, X, dists, inds, r);
    return std::make_pair(inds, dists);
}

template <typename Metric> uint64_t clusterxx::vp_tree<Metric>::depth() {
    return __depth(__root);
}

#endif
