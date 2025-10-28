#ifndef CLUSTERXX_BASE_MANIFOLD_METHOD_HPP
#define CLUSTERXX_BASE_MANIFOLD_METHOD_HPP

#include "clusterxx/writing/write_json.hpp"

#include <armadillo>

namespace clusterxx {
    /**
     * @brief base class for manifold methods
     */
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

    /**
     * @brief fit member function for manifold methods
     * @param X: the input data
     */
    virtual void fit(const arma::mat &X) = 0;
    /**
     * @brief fit_transform member function for manifold methods
     * @param X: the input data
     *
     * @return arma::mat: the latent features
     */
    virtual arma::mat fit_transform(const arma::mat &X) = 0;
    /**
     * @brief get_out_features member function for manifold methods
     *
     * @return arma::mat: the latent features from fit/fit_transform
     */
    arma::mat get_out_features() const { return __out_features; }
    /**
     * @brief get_shape memeber function for manifold methods
     *
     * @return std::pair<size_t, size_t>: shape of the latent features
     */
    std::pair<size_t, size_t> get_shape() const { return __shape; }
    /**
     * @brief save_to_json member function. Saves the results to a json file
     *
     * @param filename: the filename of the resulted json file
     */
    void save_to_json(const std::string &filename) {
        clusterxx::save_to_json(__out_features, filename);
    }
};
} // namespace clusterxx

#endif
