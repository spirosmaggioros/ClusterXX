#ifndef CLUSTERXX_DATA_STRUCTURES_GRAPH_HPP
#define CLUSTERXX_DATA_STRUCTURES_GRAPH_HPP

#include <unordered_map>
#include <vector>
#include <utility>

namespace clusterxx {
class Graph {
  private:
      std::string __type;
      std::unordered_map<unsigned int, std::vector<std::pair<unsigned int, double>>>
          __adj_list;
  public:
      Graph(const std::string &type = "undirected");

      void insert_edge(const unsigned int &u, const unsigned int &v, const double &key);
      std::vector<std::vector<double>> floyd_warshall();
};
}

#include "graph_impl.hpp"

#endif
