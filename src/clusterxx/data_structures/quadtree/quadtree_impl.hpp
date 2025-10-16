#ifndef CLUSTERXX_DATA_STRUCTURES_QUADTREE_IMPL_HPP
#define CLUSTERXX_DATA_STRUCTURES_QUADTREE_IMPL_HPP

#include "quadtree.hpp"

#include <cfloat>

template <uint8_t dim, uint32_t node_capacity>
void clusterxx::quadtree<dim, node_capacity>::__insert(
    std::unique_ptr<quadtree_node<node_type>> &node, const size_t idx) {
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

template <uint8_t dim, uint32_t node_capacity>
void clusterxx::quadtree<dim, node_capacity>::__initialize() {
    double _min_x = DBL_MAX, _max_x = DBL_MIN;
    double _min_y = DBL_MAX, _max_y = DBL_MIN;
    double _min_z = DBL_MAX, _max_z = DBL_MIN;
    _min_x = std::min(_min_x, __in_features.col(0).min());
    _max_x = std::max(_max_x, __in_features.col(0).max());
    _min_y = std::min(_min_y, __in_features.col(1).min());
    _max_y = std::max(_max_y, __in_features.col(1).max());
    if (dim == 3) {
        _min_z = std::min(_min_z, __in_features.col(2).min());
        _max_z = std::max(_max_z, __in_features.col(2).max());
    }

    AABB<node_type> _init_space;
    _init_space.__center.__x = (_min_x + _max_x) / 2;
    _init_space.__center.__y = (_min_y + _max_y) / 2;

    if (dim == 3) {
        _init_space.__center.__z = (_min_z + _max_z) / 2;
    }

    _init_space.__half_dim = _max_x - _init_space.__center.__x / 2;
    __root = std::make_unique<quadtree_node<node_type>>(_init_space);

    for (size_t i = 0; i < __in_features.n_rows; i++) {
        __insert(__root, i);
    }
}

template <uint8_t dim, uint32_t node_capacity>
clusterxx::quadtree<dim, node_capacity>::quadtree(const arma::mat &X)
    : __in_features(X) {
    assert(!X.empty());
}

#endif
