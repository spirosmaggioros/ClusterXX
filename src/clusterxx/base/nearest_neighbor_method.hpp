#ifndef CLUSTERXX_BASE_NEAREST_NEIGHBOR_METHOD_HPP
#define CLUSTERXX_BASE_NEAREST_NEIGHBOR_METHOD_HPP

namespace clusterxx {
class nearest_neighbor {
  public:
    nearest_neighbor() = default;
    virtual ~nearest_neighbor() {};
    nearest_neighbor(const nearest_neighbor &) = default;
    nearest_neighbor(nearest_neighbor &&) = default;
    nearest_neighbor &operator=(const nearest_neighbor &) = default;
    nearest_neighbor &operator=(nearest_neighbor &&) = default;
};
} // namespace clusterxx

#endif
