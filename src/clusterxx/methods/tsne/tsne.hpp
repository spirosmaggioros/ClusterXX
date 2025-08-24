#ifndef CLUSTERXX_METHODS_TSNE_HPP
#define CLUSTERXX_METHODS_TSNE_HPP

#include <assert.h>
#include <utility>
#include <vector>

#include "../../base/manifold_method.hpp"
#include "../../metrics/metrics.hpp"

namespace clusterxx {
template <typename Metric = clusterxx::metrics::euclidean_distance>
class TSNE : manifold_method {
  private:
    int __n_components;
    double __perplexity;
    double __learning_rate;
    int __max_iter;
    double __momentum = 0.5;
    Metric metric;

    std::pair<int, int> __shape;
    std::vector<std::vector<double>> __features;

    void __fit(const std::vector<std::vector<double>> &X);
    double __compute_sigma(const std::vector<double> &distances,
                           double target_perplexity, double tolerance = 1e-5,
                           int max_iter = 100);
    std::vector<std::vector<double>>
    __compute_pairwise_affinities(std::vector<std::vector<double>> features,
                                  double perplexity);
    std::vector<std::vector<double>>
    __compute_low_dim_affinities(const std::vector<std::vector<double>> Y);
    std::vector<std::vector<double>> __kullback_leibler_gradient(
        std::vector<std::vector<double>> pairwise_affinities,
        std::vector<std::vector<double>> low_dim_affinities,
        std::vector<std::vector<double>> low_dim_features);

  public:
    TSNE(int n_components = 2, double perplexity = 30.0,
         double learning_rate = 100, int max_iter = 1000)
        : __n_components(n_components), __perplexity(perplexity),
          __learning_rate(learning_rate), __max_iter(max_iter) {
        assert(n_components > 1);
        assert(__learning_rate > 0.0);
    }
    ~TSNE() {}

    void fit(const std::vector<std::vector<double>> &X) override;
    std::vector<std::vector<double>>
    fit_transform(const std::vector<std::vector<double>> &X) override;
    std::pair<int, int> get_shape();
    std::vector<std::vector<double>> get_features();
};
} // namespace clusterxx

#include "tsne_impl.hpp"

#endif
