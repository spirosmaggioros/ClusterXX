#ifndef CLUSTERXX_METHODS_TSNE_HPP
#define CLUSTERXX_METHODS_TSNE_HPP

#include <armadillo>
#include <assert.h>
#include <utility>

#include "clusterxx/base/manifold_method.hpp"
#include "clusterxx/metrics/metrics.hpp"

namespace clusterxx {
template <typename Metric = clusterxx::pairwise_distances::squared_euclidean_distances>
class TSNE : manifold_method {
  private:
    unsigned int __n_components;
    double __perplexity;
    double __learning_rate;
    double __early_exaggeration;
    unsigned int __max_iter;
    double __momentum = 0.5;
    double __min_grad_norm;
    unsigned int __n_iter_without_progress;
    Metric metric;

    std::pair<int, int> __shape;
    arma::mat __features;

    void __fit(const arma::mat &X);
    double __compute_beta(const arma::mat &distances, double target_perplexity,
                          int iter, double tolerance = 1e-5,
                          int max_iter = 200);
    arma::mat __compute_pairwise_affinities(const arma::mat &features,
                                            double perplexity);
    std::pair<arma::mat, arma::mat>
    __compute_low_dim_affinities(const arma::mat &Y);
    arma::mat __kullback_leibler_gradient(const arma::mat &pairwise_affinities,
                                          const arma::mat &low_dim_affinities,
                                          const arma::mat &low_dim_features,
                                          const arma::mat &pairwise_dists);

  public:
    TSNE(const unsigned int n_components = 2,
         const double perplexity = 30.0,
         const double learning_rate = 200,
         const double early_exaggeration = 12.0,
         const unsigned int max_iter = 1000,
         const double min_grad_norm = 1e-7,
         const unsigned int n_iter_without_progress = 300)
        : __n_components(n_components), __perplexity(perplexity),
          __learning_rate(learning_rate),
          __early_exaggeration(early_exaggeration), __max_iter(max_iter),
          __min_grad_norm(min_grad_norm),
          __n_iter_without_progress(n_iter_without_progress) {
        assert(n_components > 0);
        assert(perplexity > 0);
        assert(learning_rate > 0.0);
        assert(early_exaggeration >= 1.0);
        assert(max_iter >= 20);
        assert(min_grad_norm > 0.0);
        assert(n_iter_without_progress > 0);
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
