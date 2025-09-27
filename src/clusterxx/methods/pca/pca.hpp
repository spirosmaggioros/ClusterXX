#ifndef CLUSTERXX_METHODS_PCA_HPP
#define CLUSTERXX_METHODS_PCA_HPP

#include "clusterxx/base/decomposition_method.hpp"
#include <assert.h>

namespace clusterxx {
class PCA : decomposition_method {
  private:
    const uint16_t __n_components;
    arma::mat __signals;
    arma::vec __explained_variance;

    void __fit(const arma::mat &X);

  public:
    PCA(const uint16_t n_components) : __n_components(n_components) {}
    void fit(const arma::mat &X) override;
    arma::mat fit_transform(const arma::mat &X) override;
    arma::vec get_explained_variance() const;
};
} // namespace clusterxx

#include "pca_impl.hpp"

#endif
