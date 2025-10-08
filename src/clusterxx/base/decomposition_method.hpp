#ifndef CLUSTERXX_BASE_DECOMPOSITION_METHOD_HPP
#define CLUSTERXX_BASE_DECOMPOSITION_METHOD_HPP

#include "clusterxx/writing/write_json.hpp"

#include <armadillo>

namespace clusterxx {
class decomposition_method {
  protected:
    arma::mat __out_features;
    std::pair<size_t, size_t> __shape;
  public:
    decomposition_method() = default;
    virtual ~decomposition_method() {};
    decomposition_method(const decomposition_method &) = delete;
    decomposition_method(decomposition_method &&) = delete;
    decomposition_method &operator=(const decomposition_method &) = delete;
    decomposition_method &operator=(decomposition_method &&) = delete;

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
