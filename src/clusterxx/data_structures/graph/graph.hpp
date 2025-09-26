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

  public:
    Graph(const std::string &type = "undirected");

    void insert_edge(const unsigned int &u, const unsigned int &v,
                     const double &key);
    size_t size() const;
    std::vector<double> floyd_warshall();
};
} // namespace clusterxx

#include "graph_impl.hpp"

#endif
