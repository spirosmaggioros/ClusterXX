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
class KMeans : cluster_method {
  private:
    Metric metric;
    std::unordered_map<int, std::vector<int>> __assignments;
    arma::mat __centroids;
    std::vector<int> __labels;

    int __n_clusters;
    int __max_iter;
    std::string __init;
    arma::mat __features;

    std::optional<int> __random_state;

    void __fit(const arma::mat &X);
    void __init_centroids(arma::mat features);
    void __assign_labels(const arma::mat &X);
    arma::mat __recalc_centroids();

  public:
    KMeans(int n_clusters = 8, int max_iter = 300,
           std::string init = "k-means++",
           std::optional<int> random_state = std::nullopt)
        : __n_clusters(n_clusters), __max_iter(max_iter), __init(init),
          __random_state(random_state) {
        assert(max_iter > 0);
        assert(n_clusters > 0);
    }

    ~KMeans() {}

    void fit(const arma::mat &X) override;
    std::vector<int> fit_predict(const arma::mat &X) override;
    std::vector<int> predict(const arma::mat &X) override;
    std::vector<int> get_labels() const;
    arma::mat get_centroids() const;
};
} // namespace clusterxx

#include "kmeans_impl.hpp"

#endif
