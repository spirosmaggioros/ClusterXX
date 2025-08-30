#ifndef CLUSTERXX_METHODS_MBKMEANS_HPP
#define CLUSTERXX_METHODS_MBKMEANS_HPP

#include "clusterxx/base/cluster_method.hpp"
#include "clusterxx/metrics/metrics.hpp"
#include <armadillo>
#include <optional>

namespace clusterxx {
template <typename Metric = clusterxx::pairwise_distances::euclidean_distances>
class MiniBatchKMeans : cluster_method {
  private:
    Metric metric;
    std::unordered_map<int, std::vector<int>> __assignments;
    arma::mat __centroids;
    std::vector<int> __labels;
    std::vector<int> __center_counts;

    int __n_clusters;
    int __max_iter;
    int __batch_size;
    std::string __init;
    arma::mat __features;

    std::optional<int> __random_state;

    void __fit(const arma::mat &X);
    void __init_centroids();
    void __assign_labels(const arma::mat &X, const arma::uvec &batches);

  public:
    MiniBatchKMeans(const int n_clusters = 8,
                    const std::string init = "k-means++",
                    const int max_iter = 100, const int batch_size = 1024,
                    std::optional<int> random_state = std::nullopt)
        : __n_clusters(n_clusters), __init(init), __max_iter(max_iter),
          __batch_size(batch_size), __random_state(random_state) {
        assert(max_iter > 0);
        assert(n_clusters > 0);
        assert(init == "k-means++" || init == "random");
    }

    ~MiniBatchKMeans() {}

    void fit(const arma::mat &X) override;
    std::vector<int> fit_predict(const arma::mat &X) override;
    std::vector<int> predict(const arma::mat &X) override;
    std::vector<int> get_labels() const;
    arma::mat get_centroids() const;
};
} // namespace clusterxx

#include "mbkmeans_impl.hpp"

#endif
