#ifndef CLUSTERXX_DATA_STRUCTURES_QUADTREE_HPP
#define CLUSTERXX_DATA_STRUCTURES_QUADTREE_HPP

#include <armadillo>
#include <assert.h>
#include <memory>
#include <optional>
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
            
        __2d_point() : __x(0.0), __y(0.0) {}
        __2d_point(const double x, const double y) : __x(x), __y(y) {}
        std::pair<double, double> get_point() {
            return std::make_pair(__x, __y);
        }
    };

    struct __3d_point {
        double __x;
        double __y;
        double __z;
        
        __3d_point() : __x(0.0), __y(0.0), __z(0.0) {}
        __3d_point(const double x, const double y, const double z)
            : __x(x), __y(y), __z(z) {}
        std::tuple<double, double, double> get_point() {
            return std::make_tuple(__x, __y, __z);
        }
    };

    template <class T> struct AABB {
        T __center;
        double __half_dim;

        AABB() = default;

        bool contains_point(const arma::vec &point) {
            double _x = point(0);
            double _y = point(1);
            if constexpr (std::is_same_v<T, __2d_point>) {
                assert(point.n_rows == 2);
                auto [x, y] = __center.get_point();
                return (x - __half_dim <= _x && _x <= x + __half_dim) &&
                       (y - __half_dim <= _y && _y <= y + __half_dim);
            } else {
                assert(point.n_rows == 3);
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
                       (_y + _half_dim > y - __half_dim) ||
                       (_y - _half_dim < y + __half_dim);
;
            } else {
                auto [_x, _y, _z] = point.__center.get_point();
                double _half_dim = point.__half_dim;
                auto [x, y, z] = __center.get_point();
                return (_x + _half_dim > x - __half_dim) ||
                       (_x - _half_dim < x + __half_dim) ||
                       (_y + _half_dim > y - __half_dim) ||
                       (_y - _half_dim < y + __half_dim) ||
                       (_z + _half_dim > z - __half_dim) ||
                       (_z - _half_dim < z + __half_dim);
            }
        }
    };

    template <class T> struct quadtree_node {
        std::unique_ptr<quadtree_node> NW;
        std::unique_ptr<quadtree_node> NE;
        std::unique_ptr<quadtree_node> SW;
        std::unique_ptr<quadtree_node> SE;

        std::vector<size_t> __points;
        AABB<T> __center;

        quadtree_node(AABB<T> &center)
            : NW(nullptr), NE(nullptr), SW(nullptr), SE(nullptr),
              __center(center) {}
    
        void subdivide() {
            AABB<T> NE_center = AABB<T>();
            AABB<T> NW_center = AABB<T>();
            AABB<T> SW_center = AABB<T>();
            AABB<T> SE_center = AABB<T>();

            double NE_x_center, NE_y_center, NE_z_center;
            double NW_x_center, NW_y_center, NW_z_center;
            double SW_x_center, SW_y_center, SW_z_center;
            double SE_x_center, SE_y_center, SE_z_center;

            auto _set_new_centers = [&](double x, double y, std::optional<double> z = std::nullopt) -> void {
                NE_x_center = (2 * x + __center.__half_dim) / 2;
                NE_y_center = (2 * y + __center.__half_dim) / 2;
                NW_x_center = (2 * x - __center.__half_dim) / 2;
                NW_y_center = (2 * y + __center.__half_dim) / 2;
                SW_x_center = (2 * x - __center.__half_dim) / 2;
                SW_y_center = (2 * y - __center.__half_dim) / 2;
                SE_x_center = (2 * x + __center.__half_dim) / 2;
                SE_y_center = (2 * y - __center.__half_dim) / 2;
                if (z != std::nullopt) {
                    NE_z_center = (2 * z.value() - __center.__half_dim) / 2;
                    NW_z_center = (2 * z.value() - __center.__half_dim) / 2;
                    SW_z_center = (2 * z.value() + __center.__half_dim) / 2;
                    SE_z_center = (2 * z.value() + __center.__half_dim) / 2;
                }
            };

            if constexpr (std::is_same_v<T, __2d_point>) {
                auto [_x, _y] = __center.__center.get_point();
                _set_new_centers(_x, _y);

                NE_center.__center = T(NE_x_center, NE_y_center);
                NW_center.__center = T(NW_x_center, NW_y_center);
                SW_center.__center = T(SW_x_center, SW_y_center);
                SE_center.__center = T(SE_x_center, SE_y_center);
            } else {
                auto [_x, _y, _z] = __center.__center.get_point();
                _set_new_centers(_x, _y, _z);

                NE_center.__center = T(NE_x_center, NE_y_center, NE_z_center);
                NW_center.__center = T(NW_x_center, NW_y_center, NW_z_center);
                SE_center.__center = T(SE_x_center, SE_y_center, SE_z_center);
                SW_center.__center = T(SW_x_center, SW_y_center, SW_z_center);
            }

            NE_center.__half_dim = __center.__half_dim / 2;
            NW_center.__half_dim = NE_center.__half_dim;
            SE_center.__half_dim = NE_center.__half_dim;
            SW_center.__half_dim = NE_center.__half_dim;

            NE = std::make_unique<quadtree_node<T>>(NE_center);
            NW = std::make_unique<quadtree_node<T>>(NW_center);
            SE = std::make_unique<quadtree_node<T>>(SE_center);
            SW = std::make_unique<quadtree_node<T>>(SW_center);
        }
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
