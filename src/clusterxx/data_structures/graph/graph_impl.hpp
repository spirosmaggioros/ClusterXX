#ifndef CLUSTERXX_DATA_STRUCTURES_GRAPH_IMPL_HPP
#define CLUSTERXX_DATA_STRUCTURES_GRAPH_IMPL_HPP

#include "graph.hpp"
#include <assert.h>
#include <cfloat>
#include <queue>

clusterxx::Graph::Graph(const std::string &type) : __type(type), __edges(0) {
    assert(type == "undirected" || type == "directed");
}

void clusterxx::Graph::insert_edge(const unsigned int &u, const unsigned int &v,
                                   const double &key) {
    __adj_list[u].push_back({v, key});
    if (__type == "undirected") {
        __adj_list[v].push_back({u, key});
        __edges += 2;
    } else {
        __edges++;
    }
}

void clusterxx::Graph::remove_node(const unsigned int &u) {
    if (__adj_list.find(u) == __adj_list.end()) {
        return;
    }

    __adj_list.erase(u);
    auto find_u = [&u](const std::pair<unsigned int, double> &a) {
        return a.first == u;
    };
    for (auto &[v, N_v] : __adj_list) {
        const auto iter_u = std::find_if(N_v.begin(), N_v.end(), find_u);
        N_v.erase(iter_u);
    }
}

size_t clusterxx::Graph::n_nodes() const { return __adj_list.size(); }

size_t clusterxx::Graph::n_edges() const { return __edges; }

std::vector<double> clusterxx::Graph::floyd_warshall_all_shortest_paths() {
    size_t total_nodes = __adj_list.size();
    std::vector<double> dists(total_nodes * total_nodes, DBL_MAX);

    for (auto &[u, neigh_u] : __adj_list) {
        for (auto &[neigh, dist] : neigh_u) {
            dists[total_nodes * u + neigh] = dist;
        }
    }

    for (size_t k = 0; k < total_nodes; k++) {
        for (size_t i = 0; i < total_nodes; i++) {
            for (size_t j = 0; j < total_nodes; j++) {
                dists[total_nodes * i + j] = std::min(
                    dists[total_nodes * i + j],
                    dists[total_nodes * i + k] + dists[total_nodes * k + j]);
            }
        }
    }

    return dists;
}

std::vector<double> clusterxx::Graph::bellman_ford(const unsigned int &s) {
    size_t total_nodes = __adj_list.size();
    std::vector<double> dists(total_nodes, DBL_MAX);
    dists[s] = 0.0;

    for (size_t i = 0; i < total_nodes - 1; i++) {
        for (const auto &[u, N_u] : __adj_list) {
            for (const auto &[_u, d] : N_u) {
                if (dists[u] + d < dists[_u]) {
                    dists[_u] = dists[u] + d;
                }
            }
        }
    }
    // we will never have negative cycles for a manifold/clustering problem

    return dists;
}

std::vector<double>
clusterxx::Graph::dijkstra_from_single(const unsigned int &s) {
    size_t total_nodes = __adj_list.size();
    std::vector<double> dists(total_nodes, DBL_MAX);

    std::priority_queue<std::pair<double, unsigned int>,
                        std::vector<std::pair<double, unsigned int>>,
                        std::greater<std::pair<double, unsigned int>>>
        pq;
    pq.push({0, s});
    dists[s] = 0;

    while (!pq.empty()) {
        auto [current_dist, current_node] = pq.top();
        pq.pop();

        for (const auto &[u, d] : __adj_list[current_node]) {
            if (current_dist + d < dists[u]) {
                dists[u] = current_dist + d;
                pq.push({dists[u], u});
            }
        }
    }

    return dists;
}

std::vector<std::vector<double>>
clusterxx::Graph::dijkstra_all_shortest_paths() {
    size_t total_nodes = __adj_list.size();

    std::vector<std::vector<double>> dists;
    for (size_t i = 0; i < total_nodes; i++) {
        std::vector<double> curr_dists = dijkstra_from_single(i);
        dists.push_back(curr_dists);
    }

    return dists;
}

#endif
