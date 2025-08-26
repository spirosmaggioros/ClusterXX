#ifndef CLUSTERXX_BASE_CLUSTER_METHOD_HPP
#define CLUSTERXX_BASE_CLUSTER_METHOD_HPP

#include <vector>
#include <armadillo>

namespace clusterxx {
class cluster_method {
  public:
    cluster_method() = default;
    virtual ~cluster_method() {};
    cluster_method(const cluster_method &) = default;
    cluster_method(cluster_method &&) = default;
    cluster_method &operator=(const cluster_method &) = default;
    cluster_method &operator=(cluster_method &&) = default;

    virtual void fit(const arma::mat &X) = 0;
    virtual std::vector<int>
    fit_predict(const arma::mat &X) = 0;
    virtual std::vector<int>
    predict(const arma::mat &X) = 0;
};
} // namespace clusterxx

#endif
