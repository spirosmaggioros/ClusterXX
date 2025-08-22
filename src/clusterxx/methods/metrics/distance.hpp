#ifndef CLUSTERXX_METRICS_DISTANCE_HPP
#define CLUSTERXX_METRICS_DISTANCE_HPP

#include <vector>
#include <cassert>
#include <string.h>


std::vector<double> vector_distance(const std::vector<double> a, std::vector<double> b) {
    assert(a.size() == b.size());
    std::vector<double> d_ij(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        d_ij[i] = (a[i] - b[i]) * (a[i] - b[i]);
    }

    return d_ij;
}

#endif

