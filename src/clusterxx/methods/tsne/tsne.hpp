#ifndef CLUSTERXX_METHODS_TSNE_HPP
#define CLUSTERXX_METHODS_TSNE_HPP

#include <assert.h>
#include <utility>
#include <vector>
#include <armadillo>

#include "clusterxx/base/manifold_method.hpp"
#include "clusterxx/metrics/metrics.hpp"

namespace clusterxx {
template <typename Metric = clusterxx::pairwise_distances::euclidean_distances>
class TSNE : manifold_method {
  private:
    int __n_components;
    double __perplexity;
    double __learning_rate;
    double __early_exaggeration;
    int __max_iter;
    double __momentum = 0.5;
    Metric metric;

    std::pair<int, int> __shape;
    arma::mat __features;

    void __fit(const arma::mat &X);
    double __compute_sigma(const arma::mat &distances,
                           double target_perplexity, int iter,
                           double tolerance = 1e-5, int max_iter = 50);
    arma::mat __compute_pairwise_affinities(
        const arma::mat &features, double perplexity);
    arma::mat __compute_low_dim_affinities(const arma::mat &Y);
    arma::mat __kullback_leibler_gradient(
        const arma::mat &pairwise_affinities,
        const arma::mat &low_dim_affinities,
        const arma::mat &low_dim_features);

  public:
    TSNE(int n_components = 2, double perplexity = 30.0,
         double learning_rate = 100, double early_exaggeration = 4.0,
         int max_iter = 1000)
        : __n_components(n_components), __perplexity(perplexity),
          __learning_rate(learning_rate),
          __early_exaggeration(early_exaggeration), __max_iter(max_iter) {
        assert(n_components > 1);
        assert(perplexity > 0);
        assert(learning_rate > 0.0);
        assert(early_exaggeration >= 1.0);
        assert(max_iter >= 20);
    }
    ~TSNE() {}

    void fit(const arma::mat &X) override;
    arma::mat fit_transform(const arma::mat &X) override;
    std::pair<int, int> get_shape() const;
    arma::mat get_features() const;
};
} // namespace clusterxx

#include "tsne_impl.hpp"

#endif
