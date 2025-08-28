#ifndef CLUSTERXX_DATA_STRUCTURES_KD_TREE_HPP
#define CLUSTERXX_DATA_STRUCTURES_KD_TREE_HPP

#include "clusterxx/metrics/metrics.hpp"
#include <armadillo>
#include <assert.h>
#include <memory>
#include <queue>
#include <vector>

namespace clusterxx {
template <typename Metric = clusterxx::metrics::euclidean_distance>
class kd_tree {
  private:
    struct kd_node {
        std::unique_ptr<kd_node> left = nullptr;
        std::unique_ptr<kd_node> right = nullptr;
        arma::vec __point;
        size_t __ind;
        size_t __feature_size;

        kd_node(const arma::vec &point, const size_t ind)
            : __point(point), __feature_size(__point.n_cols), __ind(ind) {}
        void add(std::unique_ptr<kd_node> kd_node, const int depth = 0) {
            assert(kd_node->__point.n_cols == __point.n_cols);
            if (kd_node->__point(depth % __feature_size) <
                __point(depth % __feature_size)) {
                if (!left) {
                    left = std::move(kd_node);
                } else {
                    left->add(std::move(kd_node), depth + 1);
                }
            } else {
                if (!right) {
                    right = std::move(kd_node);
                } else {
                    right->add(std::move(kd_node), depth + 1);
                }
            }
        }
    };

    struct Compare {
        bool operator()(const std::pair<double, int> &a,
                        const std::pair<double, int> &b) {
            return a.first < b.first;
        }
    };
    using MaxHeap =
        std::priority_queue<std::pair<double, int>,
                            std::vector<std::pair<double, int>>, Compare>;
    std::unique_ptr<kd_node> __root;

    void __initialize();
    void __k_nearest_neighbors(std::unique_ptr<kd_node> node,
                               const arma::vec &X, MaxHeap &heap,
                               const int depth = 0, const int k = 1);
    void __query_nearest_neighbors(std::unique_ptr<kd_node> node,
                                   const arma::vec &X, MaxHeap &heap,
                                   const double radius, const int depth = 0);

    Metric metric;
    arma::mat __features;
    int __leaf_size;

  public:
    kd_tree(const arma::mat &X, const int leaf_size = 40)
        : __root(nullptr), __leaf_size(leaf_size), __features(X) {
        assert(!X.empty());
        assert(leaf_size > 0);
        __initialize();
    }

    void add(const arma::vec &feature);
    arma::mat query(const arma::vec &X, const int &k = 1);
    arma::mat query_radius(const arma::vec &X, const double &r);
};
} // namespace clusterxx

#include "kd_tree_impl.hpp"

#endif
