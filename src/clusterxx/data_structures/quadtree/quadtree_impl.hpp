#ifndef CLUSTERXX_DATA_STRUCTURES_QUADTREE_IMPL_HPP
#define CLUSTERXX_DATA_STRUCTURES_QUADTREE_IMPL_HPP

#include "quadtree.hpp"

#include <cfloat>
#include <queue>

template <uint32_t node_capacity>
void clusterxx::quadtree<node_capacity>::__insert(
    std::unique_ptr<quadtree_node> &node, const size_t idx) {
    if (!node->__center.contains_point(__in_features.row(idx).t())) {
        return;
    }

    if (!node->is_full() && node->NW == nullptr) {
        node->__points.push_back(idx);
        return;
    }

    if (node->NW == nullptr) {
        node->subdivide();
    }

    __insert(node->NW, idx);
    __insert(node->NE, idx);
    __insert(node->SW, idx);
    __insert(node->SE, idx);
}

template <uint32_t node_capacity>
void clusterxx::quadtree<node_capacity>::__initialize() {
    double _min_x = DBL_MAX, _max_x = DBL_MIN;
    double _min_y = DBL_MAX, _max_y = DBL_MIN;

    _min_x = std::min(_min_x, __in_features.col(0).min());
    _max_x = std::max(_max_x, __in_features.col(0).max());
    _min_y = std::min(_min_y, __in_features.col(1).min());
    _max_y = std::max(_max_y, __in_features.col(1).max());

    AABB _init_space = AABB();
    _init_space.__center.__x = (_min_x + _max_x) / 2;
    _init_space.__center.__y = (_min_y + _max_y) / 2;

    double max_x_axis = std::max(std::abs(_min_x), std::abs(_max_x));
    double max_y_axis = std::max(std::abs(_min_y), std::abs(_max_y));
    if (max_x_axis > max_y_axis) {
        _init_space.__half_dim = max_x_axis - _init_space.__center.__x;
    } else {
        _init_space.__half_dim = max_y_axis - _init_space.__center.__y;
    }
    __root = std::make_unique<quadtree_node>(_init_space);

    for (size_t i = 0; i < __in_features.n_rows; i++) {
        __insert(__root, i);
    }
}

template <uint32_t node_capacity>
void clusterxx::quadtree<node_capacity>::__range_query(
    std::unique_ptr<quadtree_node> &node, std::vector<size_t> &pts_in_range,
    const AABB &search_space) {
    if (!node->__center.intersects_point(search_space)) {
        return;
    }

    for (size_t i = 0; i < node->__points.size(); i++) {
        if (search_space.contains_point(
                __in_features.row(node->__points[i]).t())) {
            pts_in_range.push_back(node->__points[i]);
        }
    }

    if (!node->NW) {
        return;
    }

    __range_query(node->NW, pts_in_range, search_space);
    __range_query(node->NE, pts_in_range, search_space);
    __range_query(node->SW, pts_in_range, search_space);
    __range_query(node->SE, pts_in_range, search_space);
}

template <uint32_t node_capacity>
std::vector<size_t>
clusterxx::quadtree<node_capacity>::range_query(const arma::vec &point,
                                                const double &half_dim) {
    assert(point.n_elem == 2);
    AABB search_space = AABB(point(0), point(1), half_dim);

    std::vector<size_t> pts_in_range;
    __range_query(__root, pts_in_range, search_space);

    return pts_in_range;
}

template <uint32_t node_capacity>
uint32_t clusterxx::quadtree<node_capacity>::depth() const {
    if (!__root) {
        return 0;
    }
    uint32_t _depth = 0;
    std::queue<quadtree_node *> q;
    q.push(__root.get());

    while (!q.empty()) {
        size_t size = q.size();
        for (size_t i = 0; i < size; i++) {
            quadtree_node *curr = q.front();
            q.pop();

            if (curr->NW) {
                q.push(curr->NW.get());
            }
            if (curr->NE) {
                q.push(curr->NE.get());
            }
            if (curr->SW) {
                q.push(curr->SW.get());
            }
            if (curr->SE) {
                q.push(curr->SE.get());
            }
        }
        _depth++;
    }

    return _depth;
}

template <uint32_t node_capacity>
clusterxx::quadtree<node_capacity>::quadtree(const arma::mat &X)
    : __in_features(X) {
    assert(!X.empty());
    assert(X.n_cols == 2); // until we have a 3D quadtree ready
    __initialize();
}

#endif
