#ifndef CLUSTERXX_DATA_STRUCTURES_GRAPH_IMPL_HPP
#define CLUSTERXX_DATA_STRUCTURES_GRAPH_IMPL_HPP

#include "graph.hpp"
#include <assert.h>
#include <cfloat>

clusterxx::Graph::Graph(const std::string &type) : __type(type) {
    assert(type == "undirected" || type == "directed");
}

void clusterxx::Graph::insert_edge(const unsigned int &u, const unsigned int &v,
                                   const double &key) {
    __adj_list[u].push_back({v, key});
    if (__type == "undirected") {
        __adj_list[v].push_back({u, key});
    }
}

size_t clusterxx::Graph::size() const { return __adj_list.size(); }

std::vector<double> clusterxx::Graph::floyd_warshall() {
    size_t total_nodes = __adj_list.size();
    std::vector<double> dists(total_nodes * total_nodes, DBL_MAX);

    for (auto &[u, neigh_u] : __adj_list) {
        for (auto &[neigh, dist] : neigh_u) {
            dists[total_nodes * u + neigh] = dist;
        }
    }

    for (int k = 0; k < total_nodes; k++) {
        for (int i = 0; i < total_nodes; i++) {
            for (int j = 0; j < total_nodes; j++) {
                if (dists[total_nodes * i + j] >
                    dists[total_nodes * i + k] + dists[total_nodes * k + j]) {
                    dists[total_nodes * i + j] =
                        dists[total_nodes * i + k] + dists[total_nodes * k + j];
                }
            }
        }
    }

    return dists;
}

#endif
