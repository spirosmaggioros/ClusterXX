#ifndef CLUSTERXX_METHODS_KMEANS_PLUS_PLUS_HPP
#define CLUSTERXX_METHODS_KMEANS_PLUS_PLUS_HPP

#include <armadillo>

namespace clusterxx {
class kmeans_plus_plus {
  private:
    arma::mat __centroids;
    const uint16_t __n_clusters;

    void __fit(const arma::mat &X);

  public:
    kmeans_plus_plus(const uint32_t n_clusters);
    ~kmeans_plus_plus() {}

    void fit(const arma::mat &X);
    arma::mat get_centroids() const;
};
} // namespace clusterxx

#include "kmeans_plus_plus_impl.hpp"

#endif
