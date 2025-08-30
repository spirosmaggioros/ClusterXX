#ifndef CLUSTERXX_METHODS_DBSCAN_HPP
#define CLUSTERXX_METHODS_DBSCAN_HPP

#include "clusterxx/base/cluster_method.hpp"
#include "clusterxx/data_structures/kd_tree/kd_tree.hpp"
#include "clusterxx/metrics/metrics.hpp"
#include <armadillo>
#include <memory>
#include <unordered_map>

namespace clusterxx {
template <typename Metric = clusterxx::metrics::euclidean_distance,
          class Algorithm = clusterxx::kd_tree<Metric>>
class DBSCAN : cluster_method {
  private:
    // just for now, no copy constructor
    std::unique_ptr<Algorithm> __algorithm;
    double __eps;
    int __min_samples;
    std::unordered_map<int, int> __assignments;
    std::vector<int> __labels;

    void __fit(const arma::mat &X);

  public:
    DBSCAN(const double eps = 0.5, const int min_samples = 5)
        : __eps(eps), __min_samples(min_samples) {}

    ~DBSCAN() {}

    void fit(const arma::mat &X) override;
    std::vector<int> fit_predict(const arma::mat &X) override;
    std::vector<int> predict(const arma::mat &X) override;
    std::vector<int> get_labels() const;
};
} // namespace clusterxx

#include "dbscan_impl.hpp"

#endif
