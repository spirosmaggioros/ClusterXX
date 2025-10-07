#ifndef CLUSTERXX_VISUALIZATION_HPP
#define CLUSTERXX_VISUALIZATION_HPP

#include "clusterxx/base/cluster_method.hpp"
#include "clusterxx/base/decomposition_method.hpp"
#include "clusterxx/base/manifold_method.hpp"

template <typename T>
concept Cluster = std::derived_from<T, clusterxx::cluster_method>;

template <typename T>
concept Scatter = std::derived_from<T, clusterxx::manifold_method> ||
                  std::derived_from<T, clusterxx::decomposition_method>;

namespace clusterxx {
class Plot {
  public:
    Plot() = default;
    ~Plot() = default;

    template <Cluster T>
    void plot2d(const T &m, const std::string &title = "Clustering_Sesults",
                const std::string &xlabel = "X",
                const std::string &ylabel = "Y");

    template <Scatter T>
    void plot2d(const T &m, const std::string &title = "Results",
                const std::string &xlabel = "X",
                const std::string &ylabel = "Y",
                const std::vector<int> &labels = {});

    template <Cluster T>
    void plot3d(const T &m, const std::string &title = "Clustering_Sesults",
                const std::string &xlabel = "X",
                const std::string &ylabel = "Y",
                const std::string &zlabel = "Z");

    template <Scatter T>
    void plot3d(const T &m, const std::string &title = "Results",
                const std::string &xlabel = "X",
                const std::string &ylabel = "Y",
                const std::string &zlabel = "Z",
                const std::vector<int> &labels = {});
};
} // namespace clusterxx

#include "visualization_impl.hpp"

#endif
