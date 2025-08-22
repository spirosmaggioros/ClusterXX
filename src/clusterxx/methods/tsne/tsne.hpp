#ifndef CLUSTERXX_METHODS_TSNE_HPP
#define CLUSTERXX_METHODS_TSNE_HPP

#include <assert.h>
#include <utility>
#include <vector>

#include "../../base/manifold_method.hpp"

namespace clusterxx {
class TSNE : manifold_method {
  private:
    int __n_components;
    double __perplexity;
    double __learning_rate;
    int __max_iter;
    double __momentum = 0.5;

    std::pair<int, int> __shape;
    std::vector<std::vector<double>> __features;

    void __fit(const std::vector<std::vector<double>> &X);

  public:
    TSNE(int n_components = 2, double perplexity = 30.0,
         double learning_rate = 100, int max_iter = 1000)
        : __n_components(n_components), __perplexity(perplexity),
          __learning_rate(learning_rate), __max_iter(max_iter) {
        assert(n_components > 1);
        assert(__learning_rate > 0.0);
    }
    ~TSNE() {}

    void fit(const std::vector<std::vector<double>> X) override;
    std::vector<std::vector<double>>
    fit_transform(const std::vector<std::vector<double>> X) override;
    std::pair<int, int> get_shape();
    std::vector<std::vector<double>> get_features();
};
} // namespace clusterxx

#endif
