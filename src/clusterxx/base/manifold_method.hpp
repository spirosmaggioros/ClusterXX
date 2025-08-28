#ifndef CLUSTERXX_BASE_MANIFOLD_METHOD_HPP
#define CLUSTERXX_BASE_MANIFOLD_METHOD_HPP

#include <armadillo>

namespace clusterxx {
class manifold_method {
  public:
    manifold_method() = default;
    virtual ~manifold_method() {};
    manifold_method(const manifold_method &) = default;
    manifold_method(manifold_method &&) = default;
    manifold_method &operator=(const manifold_method &) = default;
    manifold_method &operator=(manifold_method &&) = default;

    virtual void fit(const arma::mat &X) = 0;
    virtual arma::mat fit_transform(const arma::mat &X) = 0;
};
} // namespace clusterxx

#endif
