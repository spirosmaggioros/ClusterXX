#ifndef CLUSTERXX_DATA_STRUCTURES_QUADTREE_HPP
#define CLUSTERXX_DATA_STRUCTURES_QUADTREE_HPP

#include <armadillo>
#include <assert.h>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace clusterxx {
template <uint8_t dim = 2, uint32_t node_capacity = 4> class quadtree {
  private:
    struct __2d_point {
        double __x;
        double __y;

        __2d_point(const double x, const double y) : __x(x), __y(y) {}
        std::pair<double, double> get_point() {
            return std::make_pair(__x, __y);
        }
    };

    struct __3d_point {
        double __x;
        double __y;
        double __z;

        __3d_point(const double x, const double y, const double z)
            : __x(x), __y(y), __z(z) {}
        std::tuple<double, double, double> get_point() {
            return std::make_tuple(__x, __y, __z);
        }
    };

    template <class T> struct AABB {
        T __center;
        double __half_dim;

        bool contains_point(const arma::vec &point) {
            if constexpr (std::is_same_v<T, __2d_point>) {
                assert(point.n_rows == 2);
                double _x = point(0);
                double _y = point(1);
                auto [x, y] = __center.get_point();

                return (x - __half_dim <= _x && _x <= x + __half_dim) &&
                       (y - __half_dim <= _y && _y <= y + __half_dim);
            } else { // 3d quadtree
                assert(point.n_rows == 3);
                double _x = point(0);
                double _y = point(1);
                double _z = point(2);
                auto [x, y, z] = __center.get_point();

                return (x - __half_dim <= _x && _x <= x + __half_dim) &&
                       (y - __half_dim <= _y && _y <= y + __half_dim) &&
                       (z - __half_dim <= _z && _z <= z + __half_dim);
            }
        }
        bool intersects_point(AABB &point) {
            if constexpr (std::is_same_v<T, __2d_point>) {
                auto [_x, _y] = point.__center.get_point();
                double _half_dim = point.__half_dim;
                auto [x, y] = __center.get_point();

                return (_x + _half_dim > x - __half_dim) ||
                       (_x - _half_dim < x + __half_dim) ||
                       (_y - _half_dim < y + __half_dim) ||
                       (_y + _half_dim > y - __half_dim);
            } else { // 3d quadtree
            }
        }
    };

    template <class T> struct quadtree_node {
        std::unique_ptr<quadtree_node> NW;
        std::unique_ptr<quadtree_node> NE;
        std::unique_ptr<quadtree_node> SW;
        std::unique_ptr<quadtree_node> SE;

        std::vector<size_t> __points;
        T __center;

        quadtree_node(T &center)
            : NW(nullptr), NE(nullptr), SW(nullptr), SE(nullptr),
              __center(center) {}

        bool is_full() { return __points.size() >= node_capacity; }
    };

    using node_type = std::conditional_t<dim == 2, __2d_point, __3d_point>;
    std::unique_ptr<quadtree_node<node_type>> __root;

    const arma::mat __in_features;

    void __insert(std::unique_ptr<quadtree_node<node_type>> &node,
                  const size_t idx);
    void __initialize();

  public:
    quadtree(const arma::mat &X);
    ~quadtree() {}

    uint32_t depth() const;
};
} // namespace clusterxx

#include "quadtree_impl.hpp"

#endif
