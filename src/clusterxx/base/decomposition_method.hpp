#ifndef CLUSTERXX_BASE_DECOMPOSITION_METHOD_HPP
#define CLUSTERXX_BASE_DECOMPOSITION_METHOD_HPP

#include <armadillo>

namespace clusterxx {
class decomposition_method {
  public:
    decomposition_method() = default;
    virtual ~decomposition_method() {};
    decomposition_method(const decomposition_method &) = default;
    decomposition_method(decomposition_method &&) = default;
    decomposition_method &operator=(const decomposition_method &) = default;
    decomposition_method &operator=(decomposition_method &&) = default;

    virtual void fit(const arma::mat &X) = 0;
    virtual arma::mat fit_transform(const arma::mat &X) = 0;
};
} // namespace clusterxx

#endif
