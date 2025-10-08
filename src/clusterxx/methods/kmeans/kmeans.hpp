#ifndef CLUSTERXX_METHODS_KMEANS_HPP
#define CLUSTERXX_METHODS_KMEANS_HPP

#include "clusterxx/base/cluster_method.hpp"
#include "clusterxx/metrics/metrics.hpp"

#include <armadillo>
#include <assert.h>
#include <optional>
#include <unordered_map>
#include <vector>

namespace clusterxx {
template <typename Metric = clusterxx::pairwise_distances::euclidean_distances>
class KMeans : public cluster_method {
  private:
    Metric metric;
    std::unordered_map<int, std::vector<int>> __assignments;
    arma::mat __centroids;

    const uint16_t __n_clusters;
    const uint32_t __max_iter;
    std::string __init;
    arma::mat __features;

    std::optional<int> __random_state;

    void __fit(const arma::mat &X);
    void __init_centroids();
    void __assign_labels(const arma::mat &X);
    arma::mat __recalc_centroids();

  public:
    KMeans(const uint16_t n_clusters = 8, const uint32_t max_iter = 300,
           std::string init = "k-means++",
           std::optional<int> random_state = std::nullopt);
    ~KMeans() {}

    void fit(const arma::mat &X) override;
    std::vector<int> fit_predict(const arma::mat &X) override;
    std::vector<int> predict(const arma::mat &X) override;
    arma::mat get_centroids() const;
};
} // namespace clusterxx

#include "kmeans_impl.hpp"

#endif
