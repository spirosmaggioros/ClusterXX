#ifndef CLUSTERXX_DATA_STRUCTURES_KD_TREE_HPP
#define CLUSTERXX_DATA_STRUCTURES_KD_TREE_HPP

#include "clusterxx/metrics/metrics.hpp"
#include <armadillo>
#include <assert.h>
#include <memory>
#include <queue>
#include <vector>

namespace clusterxx {
template <typename Metric = clusterxx::metrics::euclidean_distance,
          typename PairwiseMetric =
              clusterxx::pairwise_distances::euclidean_distances>
class kd_tree {
  private:
    struct kd_node {
        std::unique_ptr<kd_node> left;
        std::unique_ptr<kd_node> right;
        arma::vec __point;
        arma::mat __extra_points;
        std::vector<int> __extra_points_inds;
        size_t __ind;

        kd_node(const arma::vec &point, const size_t ind)
            : left(nullptr), right(nullptr), __point(point), __ind(ind) {}
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

    std::unique_ptr<kd_node> __initialize(const arma::mat &X,
                                          std::vector<size_t> &indices,
                                          int depth = 0);
    void __k_nearest_neighbors(std::unique_ptr<kd_node> &node,
                               const arma::vec &X, MaxHeap &heap,
                               const int depth = 0, const uint32_t k = 1);
    void __radius_nearest_neighbors(std::unique_ptr<kd_node> &node,
                                    const arma::vec &X,
                                    std::vector<double> &dists,
                                    std::vector<int> &inds, const double radius,
                                    const int depth = 0);
    uint32_t __depth(std::unique_ptr<kd_node> &root);
    Metric metric;
    PairwiseMetric pairwise_metric;
    unsigned int __leaf_size;

  public:
    kd_tree(const arma::mat &X, const uint16_t leaf_size = 40);
    std::pair<std::vector<int>, std::vector<double>> query(const arma::vec &X,
                                                           const int &k = 1);
    std::pair<std::vector<int>, std::vector<double>>
    query_radius(const arma::vec &X, const double &r);
    uint32_t depth();
};
} // namespace clusterxx

#include "kd_tree_impl.hpp"

#endif
