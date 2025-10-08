#ifndef CLUSTERXX_BASE_MANIFOLD_METHOD_HPP
#define CLUSTERXX_BASE_MANIFOLD_METHOD_HPP

#include "clusterxx/writing/write_json.hpp"

#include <armadillo>

namespace clusterxx {
class manifold_method {
  protected:
    arma::mat __out_features;
    std::pair<size_t, size_t> __shape;
  public:
    manifold_method() = default;
    virtual ~manifold_method() {};
    manifold_method(const manifold_method &) = delete;
    manifold_method(manifold_method &&) = delete;
    manifold_method &operator=(const manifold_method &) = delete;
    manifold_method &operator=(manifold_method &&) = delete;

    virtual void fit(const arma::mat &X) = 0;
    virtual arma::mat fit_transform(const arma::mat &X) = 0;
    arma::mat get_out_features() const { return __out_features; }
    std::pair<size_t, size_t> get_shape() const { return __shape; }
    void save_to_json(const std::string &filename) {
        clusterxx::save_to_json(__out_features, filename);
    }
};
} // namespace clusterxx

#endif
