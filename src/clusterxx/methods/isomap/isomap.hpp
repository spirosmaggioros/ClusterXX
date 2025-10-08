#ifndef CLUSTERXX_METHODS_ISOMAP_HPP
#define CLUSTERXX_METHODS_ISOMAP_HPP

#include "clusterxx/base/manifold_method.hpp"
#include "clusterxx/data_structures/kd_tree/kd_tree.hpp"
#include <memory>

namespace clusterxx {
template <class NeighAlgorithm = clusterxx::kd_tree<>>
class isomap : public clusterxx::manifold_method {
  private:
    const uint16_t __n_neighbors;
    const double __radius;
    const uint16_t __n_components;
    const std::string __path_method;
    std::unique_ptr<NeighAlgorithm> __neigh_algorithm;

    void __fit(const arma::mat &X);

  public:
    isomap(const uint16_t &n_neighbors = 5, const double &radius = 0.0,
           const uint16_t &n_components = 2,
           const std::string &path_method = "auto");
    ~isomap() {}
    void fit(const arma::mat &X) override;
    arma::mat fit_transform(const arma::mat &X) override;
};
} // namespace clusterxx

#include "isomap_impl.hpp"

#endif
