#ifndef CLUSTERXX_DATA_STRUCTURES_GRAPH_HPP
#define CLUSTERXX_DATA_STRUCTURES_GRAPH_HPP

#include <unordered_map>
#include <utility>
#include <vector>

namespace clusterxx {
class Graph {
  private:
    std::string __type;
    std::unordered_map<unsigned int,
                       std::vector<std::pair<unsigned int, double>>>
        __adj_list;
    int64_t __edges;

  public:
    Graph(const std::string &type = "undirected");

    void insert_edge(const unsigned int &u, const unsigned int &v,
                     const double &key);
    void remove_node(const unsigned int &u);
    size_t n_nodes() const;
    size_t n_edges() const;
    std::vector<double> floyd_warshall_all_shortest_paths();
    std::vector<double> bellman_ford(const unsigned int &s);
    std::vector<double> dijkstra_from_single(const unsigned int &s);
    std::vector<std::vector<double>> dijkstra_all_shortest_paths();
};
} // namespace clusterxx

#include "graph_impl.hpp"

#endif
