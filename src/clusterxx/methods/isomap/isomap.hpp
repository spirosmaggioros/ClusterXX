#ifndef CLUSTERXX_METHODS_ISOMAP_HPP
#define CLUSTERXX_METHODS_ISOMAP_HPP

#include "clusterxx/base/manifold_method.hpp"
#include "clusterxx/data_structures/kd_tree/kd_tree.hpp"
#include "clusterxx/metrics/metrics.hpp"
#include <memory>

namespace clusterxx {
template <typename Metric = clusterxx::pairwise_distances::euclidean_distances,
          class NeighAlgorithm = clusterxx::kd_tree<Metric>>
class isomap : clusterxx::manifold_method {
  private:
    const unsigned int __n_neighbors;
    const double __radius;
    const unsigned int __n_components;
    Metric metric;
    // just for now, no copy constructor
    std::unique_ptr<NeighAlgorithm> __neigh_algorithm;

  public:
    isomap(const unsigned int &n_neighbors = 5, const double &radius = 0.0,
           const unsigned int &n_components = 2);
    ~isomap() {}
    void fit(const arma::mat &X) override;
    arma::mat fit_transform(const arma::mat &X) override;
    std::pair<int, int> get_shape() const;
    arma::mat get_features() const;
};
} // namespace clusterxx

#endif
