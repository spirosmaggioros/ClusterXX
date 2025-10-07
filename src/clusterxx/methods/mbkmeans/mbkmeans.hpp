#ifndef CLUSTERXX_METHODS_MBKMEANS_HPP
#define CLUSTERXX_METHODS_MBKMEANS_HPP

#include "clusterxx/base/cluster_method.hpp"
#include "clusterxx/metrics/metrics.hpp"
#include <armadillo>
#include <optional>

namespace clusterxx {
template <typename Metric = clusterxx::pairwise_distances::euclidean_distances>
class MiniBatchKMeans : public cluster_method {
  private:
    Metric metric;
    std::unordered_map<int, std::vector<int>> __assignments;
    arma::mat __centroids;
    std::vector<int> __center_counts;

    const uint16_t __n_clusters;
    const uint32_t __max_iter;
    const uint32_t __batch_size;
    std::string __init;
    arma::mat __features;

    std::optional<int> __random_state;

    void __fit(const arma::mat &X);
    void __init_centroids();
    void __assign_labels(const arma::mat &X, const arma::uvec &batches);

  public:
    MiniBatchKMeans(const uint16_t n_clusters = 8,
                    const std::string init = "k-means++",
                    const uint32_t max_iter = 100,
                    const uint32_t batch_size = 1024,
                    std::optional<int> random_state = std::nullopt);
    ~MiniBatchKMeans() {}

    void fit(const arma::mat &X) override;
    std::vector<int> fit_predict(const arma::mat &X) override;
    std::vector<int> predict(const arma::mat &X) override;
    arma::mat get_centroids() const;
};
} // namespace clusterxx

#include "mbkmeans_impl.hpp"

#endif
