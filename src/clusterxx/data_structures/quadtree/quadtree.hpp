#ifndef CLUSTERXX_DATA_STRUCTURES_QUADTREE_HPP
#define CLUSTERXX_DATA_STRUCTURES_QUADTREE_HPP

#include <armadillo>
#include <assert.h>
#include <memory>
#include <utility>
#include <vector>

namespace clusterxx {
template <uint32_t node_capacity = 4> class quadtree {
  private:
    struct __2d_point {
        double __x;
        double __y;

        __2d_point() : __x(0.0), __y(0.0) {}
        __2d_point(const double x, const double y) : __x(x), __y(y) {}
        std::pair<double, double> get_point() const {
            return std::make_pair(__x, __y);
        }

        friend std::ostream &operator<<(std::ostream &out,
                                        const __2d_point &point) {
            out << '[' << point.__x << ", " << point.__y << "]";
            return out;
        }
    };

    struct AABB {
        __2d_point __center;
        double __half_dim;

        AABB() {
            __center.__x = 0.0;
            __center.__y = 0.0;
            __half_dim = 0.0;
        }

        AABB(const double &x, const double &y, const double &half_dim) {
            __center.__x = x;
            __center.__y = y;
            __half_dim = half_dim;
        }

        bool contains_point(const arma::vec &point) const {
            double _x = point(0);
            double _y = point(1);
            auto [x, y] = __center.get_point();
            return (x - __half_dim <= _x && _x <= x + __half_dim) &&
                   (y - __half_dim <= _y && _y <= y + __half_dim);
        }

        bool intersects_point(const AABB &point) const {
            double dx = std::abs(__center.__x - point.__center.__x);
            double dy = std::abs(__center.__y - point.__center.__y);

            return (dx <= (__half_dim + point.__half_dim)) &&
                   (dy <= (__half_dim + point.__half_dim));
        }

        friend std::ostream &operator<<(std::ostream &out, const AABB &a) {
            out << a.__center << ", half_dim: " << a.__half_dim << '\n';
            return out;
        }
    };

    struct quadtree_node {
        std::unique_ptr<quadtree_node> NW;
        std::unique_ptr<quadtree_node> NE;
        std::unique_ptr<quadtree_node> SW;
        std::unique_ptr<quadtree_node> SE;

        std::vector<size_t> __points;
        AABB __center;

        quadtree_node(AABB &center)
            : NW(nullptr), NE(nullptr), SW(nullptr), SE(nullptr),
              __center(center) {}

        void subdivide() {
            AABB NE_center = AABB();
            AABB NW_center = AABB();
            AABB SW_center = AABB();
            AABB SE_center = AABB();

            double NE_x_center, NE_y_center;
            double NW_x_center, NW_y_center;
            double SW_x_center, SW_y_center;
            double SE_x_center, SE_y_center;

            auto _set_new_centers = [&](double x, double y) -> void {
                NE_x_center = (2 * x + __center.__half_dim) / 2;
                NE_y_center = (2 * y + __center.__half_dim) / 2;
                NW_x_center = (2 * x - __center.__half_dim) / 2;
                NW_y_center = (2 * y + __center.__half_dim) / 2;
                SW_x_center = (2 * x - __center.__half_dim) / 2;
                SW_y_center = (2 * y - __center.__half_dim) / 2;
                SE_x_center = (2 * x + __center.__half_dim) / 2;
                SE_y_center = (2 * y - __center.__half_dim) / 2;
            };

            auto [_x, _y] = __center.__center.get_point();
            _set_new_centers(_x, _y);

            NE_center.__center = __2d_point(NE_x_center, NE_y_center);
            NW_center.__center = __2d_point(NW_x_center, NW_y_center);
            SW_center.__center = __2d_point(SW_x_center, SW_y_center);
            SE_center.__center = __2d_point(SE_x_center, SE_y_center);

            NE_center.__half_dim = __center.__half_dim / 2;
            NW_center.__half_dim = NE_center.__half_dim;
            SE_center.__half_dim = NE_center.__half_dim;
            SW_center.__half_dim = NE_center.__half_dim;

            NE = std::make_unique<quadtree_node>(NE_center);
            NW = std::make_unique<quadtree_node>(NW_center);
            SE = std::make_unique<quadtree_node>(SE_center);
            SW = std::make_unique<quadtree_node>(SW_center);
        }
        bool is_full() { return __points.size() >= node_capacity; }
    };

    std::unique_ptr<quadtree_node> __root;
    const arma::mat __in_features;

    void __insert(std::unique_ptr<quadtree_node> &node, const size_t idx);
    void __initialize();
    void __range_query(std::unique_ptr<quadtree_node> &node,
                       std::vector<size_t> &pts_in_range,
                       const AABB &search_space);
    void __barnes_hut_range_query(std::unique_ptr<quadtree_node> &node,
                                  const double &theta,
                                  std::vector<size_t> &pts_in_range,
                                  __2d_point &node_center,
                                  const AABB &search_space);

  public:
    quadtree(const arma::mat &X);
    ~quadtree() {}

    std::vector<size_t> range_query(const arma::vec &point,
                                    const double &half_dim);
    std::pair<std::vector<size_t>, std::pair<double, double>>
    barnes_hut_range_query(const arma::vec &point, const double &theta);
    uint32_t depth() const;
};
} // namespace clusterxx

#include "quadtree_impl.hpp"

#endif
