#ifndef CLUSTERXX_DATA_STRUCTURES_GRAPH_IMPL_HPP
#define CLUSTERXX_DATA_STRUCTURES_GRAPH_IMPL_HPP

#include "graph.hpp"
#include <assert.h>

clusterxx::Graph::Graph(const std::string &type)
    : __type(type) {
        assert(type == "undirected" || type == "directed");
}


void clusterxx::Graph::insert_edge(
        const unsigned int &u,
        const unsigned int &v,
        const double &key
) {
    __adj_list[u].push_back({v, key});
    if (__type == "undirected") {
        __adj_list[v].push_back({u, key});
    }
}

std::vector<std::vector<double>>
clusterxx::Graph::floyd_warshall() {
    size_t total_nodes = __adj_list.size();
    std::vector<std::vector<double>>
        dists(total_nodes, std::vector<double>(total_nodes, 0.0));

    for (auto &[u, neigh_u]: __adj_list) {
        for (auto &[neigh, dist]: neigh_u) {
            dists[u][neigh] = dist;
        }
    }

    for (int k = 0; k < total_nodes; k++) {
        for (int i = 0; i < total_nodes; i++) {
            for (int j = 0; j < total_nodes; j++) {
                if (dists[i][j] > dists[i][k] + dists[k][j]) {
                    dists[i][j] = dists[i][k] + dists[k][j];
                }
            }
        }
    }

    return dists;
}

#endif
