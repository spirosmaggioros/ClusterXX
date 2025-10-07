#ifndef CLUSTERXX_VISUALIZATION_HPP
#define CLUSTERXX_VISUALIZATION_HPP

#include "clusterxx/base/cluster_method.hpp"

namespace clusterxx {
class Plot {
  public:
    Plot() = default;
    ~Plot() = default;

    void plot2d(const cluster_method &m, const std::string &title = "Clustering results",
            const std::string &xlabel = "X", const std::string &ylabel = "Y");

    template <typename T>
    void plot2d(const T &m, const std::string &title = "Results",
        const std::string &xlabel = "X", const std::string &ylabel = "Y",
        const std::vector<int> &labels = {});

    void plot3d(const cluster_method &m, const std::string &title = "Clustering results",
            const std::string &xlabel = "X", const std::string &ylabel = "Y", const std::string &zlabel = "Z");

    template <typename T>
    void plot3d(const T &m, const std::string &title = "Results",
            const std::string &xlabel = "X", const std::string &ylabel = "Y",
            const std::string &zlabel = "Z", const std::vector<int> &labels = {});

};
} // namespace clusterxx

#include "visualization_impl.hpp"

#endif
