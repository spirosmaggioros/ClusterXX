#ifndef CLUSTERXX_VISUALIZATION_HPP
#define CLUSTERXX_VISUALIZATION_HPP

#include "clusterxx/base/cluster_method.hpp"
#include "clusterxx/base/decomposition_method.hpp"
#include "clusterxx/base/manifold_method.hpp"

namespace clusterxx {
class Plot {
  private:
      void scatter_plot();
  public:
      Plot() = default;
      ~Plot() = default;

      void plot(const cluster_method &m);
      void plot(const decomposition_method &m);
      void plot(const manifold_method &m);
};
}

#include "visualization_impl.hpp"

#endif
