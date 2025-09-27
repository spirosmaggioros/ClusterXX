#ifndef CLUSTERXX_DATA_STRUCTURES_KD_TREE_IMPL_HPP
#define CLUSTERXX_DATA_STRUCTURES_KD_TREE_IMPL_HPP

#include "kd_tree.hpp"

template <typename Metric, typename PairwiseMetric>
std::unique_ptr<typename clusterxx::kd_tree<Metric, PairwiseMetric>::kd_node>
clusterxx::kd_tree<Metric, PairwiseMetric>::__initialize(
    const arma::mat &X, std::vector<size_t> &indices, int depth) {
    if (indices.empty()) {
        return nullptr;
    }

    if (indices.size() <= __leaf_size) {
        auto leaf =
            std::make_unique<kd_node>(X.row(indices[0]).t(), indices[0]);
        leaf->__extra_points.resize(indices.size() - 1, X.n_cols);
        for (size_t i = 1; i < indices.size(); i++) {
            leaf->__extra_points.row(i - 1) = X.row(indices[i]);
            leaf->__extra_points_inds.push_back(indices[i]);
        }

        return leaf;
    }

    int axis = depth % X.n_cols;

    std::vector<size_t> _idx_copy = indices;
    size_t mid = indices.size() / 2;
    std::nth_element(
        _idx_copy.begin(), _idx_copy.begin() + mid, _idx_copy.end(),
        [&](size_t a, size_t b) { return X.row(a)(axis) < X.row(b)(axis); });

    size_t median = _idx_copy[mid];
    auto _nn = std::make_unique<kd_node>(X.row(median).t(), median);

    std::vector<size_t> _left(_idx_copy.begin(), _idx_copy.begin() + mid);
    std::vector<size_t> _right(_idx_copy.begin() + mid + 1, _idx_copy.end());

    _nn->left = __initialize(X, _left, depth + 1);
    _nn->right = __initialize(X, _right, depth + 1);

    return _nn;
}

template <typename Metric, typename PairwiseMetric>
void clusterxx::kd_tree<Metric, PairwiseMetric>::__k_nearest_neighbors(
    std::unique_ptr<kd_node> &node, const arma::vec &X, MaxHeap &heap,
    const int depth, const int k) {
    if (!node) {
        return;
    }

    int axis = depth % X.n_rows;
    double dist = metric(X, node->__point);

    auto emplace_heap = [&heap, &k](const double &_dist,
                                           const int &_ind) -> void {
        if (heap.size() < k) {
            heap.emplace(_dist, _ind);
        } else if (_dist < heap.top().first) {
            heap.pop();
            heap.emplace(_dist, _ind);
        }
    };
    emplace_heap(dist, node->__ind);

    if (!node->__extra_points.empty()) {
        arma::mat _X = X.t();
        arma::mat pairwise_dists = pairwise_metric(_X, node->__extra_points);
        assert(pairwise_dists.n_cols == node->__extra_points_inds.size());
        for (size_t i = 0; i < pairwise_dists.n_cols; i++) {
            double _dist = pairwise_dists(0, i);
            emplace_heap(_dist, node->__extra_points_inds[i]);
        }
    }

    std::unique_ptr<kd_node> &_next_node =
        (X(axis) < node->__point(axis)) ? node->left : node->right;
    std::unique_ptr<kd_node> &_to_check =
        (X(axis) < node->__point(axis)) ? node->right : node->left;

    __k_nearest_neighbors(_next_node, X, heap, depth + 1, k);

    double diff = std::abs(X(axis) - node->__point(axis));
    if (heap.size() < k || diff <= heap.top().first) {
        __k_nearest_neighbors(_to_check, X, heap, depth + 1, k);
    }
}

template <typename Metric, typename PairwiseMetric>
void clusterxx::kd_tree<Metric, PairwiseMetric>::__radius_nearest_neighbors(
    std::unique_ptr<kd_node> &node, const arma::vec &X,
    std::vector<double> &dists, std::vector<int> &inds, const double radius,
    const int depth) {
    if (!node) {
        return;
    }

    int axis = depth % X.n_rows;
    double dist = metric(X, node->__point);

    if (dist <= radius) {
        dists.push_back(dist);
        inds.push_back(node->__ind);
    }
    if (!node->__extra_points.empty()) {
        arma::mat _X = X.t();
        arma::mat pairwise_dists = pairwise_metric(_X, node->__extra_points);
        assert(pairwise_dists.n_cols == node->__extra_points_inds.size());
        for (size_t i = 0; i < pairwise_dists.n_cols; i++) {
            double _dist = pairwise_dists(0, i);
            if (_dist <= radius) {
                dists.push_back(_dist);
                inds.push_back(node->__extra_points_inds[i]);
            }
        }
    }

    std::unique_ptr<kd_node> &_next_node =
        (X(axis) < node->__point(axis)) ? node->left : node->right;
    std::unique_ptr<kd_node> &_to_check =
        (X(axis) < node->__point(axis)) ? node->right : node->left;

    __radius_nearest_neighbors(_next_node, X, dists, inds, radius, depth + 1);

    double diff = std::abs(X(axis) - node->__point(axis));
    if (diff <= radius) {
        __radius_nearest_neighbors(_to_check, X, dists, inds, radius,
                                   depth + 1);
    }
}

template <typename Metric, typename PairwiseMetric>
int clusterxx::kd_tree<Metric, PairwiseMetric>::__depth(
    std::unique_ptr<kd_node> &root) {
    if (!root) {
        return 0;
    }
    return 1 + std::max(__depth(root->left), __depth(root->right));
}

template <typename Metric, typename PairwiseMetric>
clusterxx::kd_tree<Metric, PairwiseMetric>::kd_tree(const arma::mat &X, const unsigned int leaf_size)
    : __leaf_size(leaf_size) {
        assert(!X.empty());
        assert(leaf_size > 0);
        assert(metric.p() > 0 && metric.p() <= 2);
        std::vector<size_t> indices(X.n_rows);
        std::iota(indices.begin(), indices.end(), 0);
        __root = __initialize(X, indices);
        // assert(depth() <= std::log2(std::max(1, (int(X.n_rows) - 1) /
        // __leaf_size)));
}

template <typename Metric, typename PairwiseMetric>
std::pair<std::vector<int>, std::vector<double>>
clusterxx::kd_tree<Metric, PairwiseMetric>::query(const arma::vec &X,
                                                  const int &k) {
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

template <typename Metric, typename PairwiseMetric>
std::pair<std::vector<int>, std::vector<double>>
clusterxx::kd_tree<Metric, PairwiseMetric>::query_radius(const arma::vec &X,
                                                         const double &r) {
    std::vector<double> dists;
    std::vector<int> inds;
    __radius_nearest_neighbors(__root, X, dists, inds, r);

    return std::make_pair(inds, dists);
}

template <typename Metric, typename PairwiseMetric>
int clusterxx::kd_tree<Metric, PairwiseMetric>::depth() {
    return __depth(__root);
}

#endif
