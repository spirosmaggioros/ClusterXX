#ifndef CLUSTERXX_BASE_MANIFOLD_METHOD_HPP
#define CLUSTERXX_BASE_MANIFOLD_METHOD_HPP

#include <vector>

namespace clusterxx {
class manifold_method {
  public:
    manifold_method() = default;
    virtual ~manifold_method() {};
    manifold_method(const manifold_method &) = default;
    manifold_method(manifold_method &&) = default;
    manifold_method &operator=(const manifold_method &) = default;
    manifold_method &operator=(manifold_method &&) = default;

    virtual void fit(const std::vector<std::vector<double>> &X) = 0;
    virtual std::vector<std::vector<double>>
    fit_transform(const std::vector<std::vector<double>> &X) = 0;
};
} // namespace clusterxx

#endif
