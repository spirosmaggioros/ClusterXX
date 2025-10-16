#ifndef CLUSTERXX_DATA_STRUCTURES_VP_TREE_HPP
#define CLUSTERXX_DATA_STRUCTURES_VP_TREE_HPP

#include "clusterxx/metrics/metrics.hpp"

#include <armadillo>
#include <cfloat>
#include <memory>
#include <queue>
#include <vector>

namespace clusterxx {
template <typename Metric = clusterxx::metrics::euclidean_distance>
class vp_tree {
  private:
    struct vp_node {
        std::unique_ptr<vp_node> left;
        std::unique_ptr<vp_node> right;

        double __mu;
        size_t __index;

        vp_node() : left(nullptr), right(nullptr), __mu(0.0) {}
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
    void __k_nearest_neighbors(std::unique_ptr<vp_node> &node,
                               const arma::vec &x, MaxHeap &heap,
                               const uint32_t &k, double &tau);
    void __radius_nearest_neighbors(std::unique_ptr<vp_node> &node,
                                    const arma::vec &X,
                                    std::vector<double> &dists,
                                    std::vector<int> &inds,
                                    const double &radius);
    uint64_t __depth(std::unique_ptr<vp_node> &root);

    std::unique_ptr<vp_node> __initialize(std::vector<size_t> &indices);

    Metric metric;
    const arma::mat __in_features;
    std::unique_ptr<vp_node> __root;

  public:
    vp_tree(const arma::mat &X);
    std::pair<std::vector<int>, std::vector<double>>
    query(const arma::vec &X, const uint32_t &k = 1);
    std::pair<std::vector<int>, std::vector<double>>
    query_radius(const arma::vec &X, const double &r);
    uint64_t depth();
};
} // namespace clusterxx

#include "vp_tree_impl.hpp"

#endif
