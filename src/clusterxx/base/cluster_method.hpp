#ifndef CLUSTERXX_BASE_CLUSTER_METHOD_HPP
#define CLUSTERXX_BASE_CLUSTER_METHOD_HPP

#include "clusterxx/writing/write_json.hpp"

#include <armadillo>
#include <vector>

namespace clusterxx {
class cluster_method {
  protected:
    std::vector<int> __labels;
    arma::mat __in_features;

  public:
    cluster_method() = default;
    virtual ~cluster_method() {};
    cluster_method(const cluster_method &) = delete;
    cluster_method(cluster_method &&) = delete;
    cluster_method &operator=(const cluster_method &) = delete;
    cluster_method &operator=(cluster_method &&) = delete;

    virtual void fit(const arma::mat &X) = 0;
    virtual std::vector<int> fit_predict(const arma::mat &X) = 0;
    virtual std::vector<int> predict(const arma::mat &X) = 0;
    arma::mat get_in_features() const { return __in_features; }
    std::vector<int> get_labels() const { return __labels; }
    void save_to_json(const std::string &filename) {
        clusterxx::save_to_json_clustering(__in_features, __labels, filename);
    }
};
} // namespace clusterxx

#endif
