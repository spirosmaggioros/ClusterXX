#ifndef CLUSTERXX_METHODS_KMEANS_HPP
#define CLUSTERXX_METHODS_KMEANS_HPP

#include "../../base/cluster_method.hpp"
#include "../../metrics/metrics.hpp"

#include <assert.h>
#include <optional>
#include <unordered_map>
#include <vector>

namespace clusterxx {
// Now this class will only support random initialization, i will implement
// kmeans++ some day
template <typename Metric = clusterxx::metrics::euclidean_distance>
class KMeans : cluster_method {
  private:
    Metric metric;
    std::unordered_map<int, std::vector<int>> __assignments;
    std::vector<std::vector<double>> __centroids;
    std::vector<int> __labels;

    int __n_clusters;
    int __max_iter;
    std::vector<std::vector<double>> __features;

    std::optional<int> __random_state;

    void __fit(const std::vector<std::vector<double>> &X);
    void __init_centroids(std::vector<std::vector<double>> features);
    void __assign_labels(const std::vector<std::vector<double>> &X);
    std::vector<std::vector<double>> __recalc_centroids();

  public:
    KMeans(int n_clusters = 8, int max_iter = 300,
           std::optional<int> random_state = std::nullopt)
        : __n_clusters(n_clusters), __max_iter(max_iter),
          __random_state(random_state) {
        assert(max_iter > 0);
        assert(n_clusters > 0);
    }

    ~KMeans() {}

    void fit(const std::vector<std::vector<double>> &X) override;
    std::vector<int>
    fit_predict(const std::vector<std::vector<double>> &X) override;
    std::vector<int>
    predict(const std::vector<std::vector<double>> &X) override;
    std::vector<int> get_labels();
    std::vector<std::vector<double>> get_centroids();
};
} // namespace clusterxx

#include "kmeans_impl.hpp"

#endif
