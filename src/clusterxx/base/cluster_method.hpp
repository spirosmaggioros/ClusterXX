#ifndef CLUSTERXX_BASE_CLUSTER_METHOD_HPP
#define CLUSTERXX_BASE_CLUSTER_METHOD_HPP

#include "clusterxx/writing/write_json.hpp"

#include <armadillo>
#include <vector>

namespace clusterxx {
    /**
     * @brief base class for clustering methods
     */
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

    /**
     * @brief fit member function for clustering methods
     * @param X: the input data
     */
    virtual void fit(const arma::mat &X) = 0;
    /**
     * @brief fit_predict member function for clustering methods
     * @param X: the input data
     *
     * @return std::vector<int>: the clustering labels
     */
    virtual std::vector<int> fit_predict(const arma::mat &X) = 0;
    /**
     * @brief predict member function for clustering methods.
     *        fit or fit_predict must be called first
     * @param X: the input data for inference
     *
     * @return std::vector<int>: the clustering labels
     */
    virtual std::vector<int> predict(const arma::mat &X) = 0;
    /**
     * @brief get_in_features member function for clustering methods
     *
     * @return arma::mat: features used for fit or fit_predict
     */
    arma::mat get_in_features() const { return __in_features; }
    /**
     * @brief get_labels member function for clustering methods
     *
     * @return std::vector<int>: labels of fit/fit_predict methods
     */
    std::vector<int> get_labels() const { return __labels; }
    /**
     * @brief save_to_json member function. Saves the results to a json file
     *
     * @param filename: the filename of the resulted json file
     */
    void save_to_json(const std::string &filename) {
        clusterxx::save_to_json(__in_features, filename, __labels);
    }
};
} // namespace clusterxx

#endif
