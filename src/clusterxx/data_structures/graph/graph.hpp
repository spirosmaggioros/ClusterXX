#ifndef CLUSTERXX_DATA_STRUCTURES_GRAPH_HPP
#define CLUSTERXX_DATA_STRUCTURES_GRAPH_HPP

#include <unordered_map>
#include <utility>
#include <vector>

namespace clusterxx {
    /**
     * @brief Simple Graph/DiGraph class
     */
class Graph {
  private:
    std::string __type;
    std::unordered_map<unsigned int,
                       std::vector<std::pair<unsigned int, double>>>
        __adj_list;
    int64_t __edges;

  public:
  /**
   * @brief Default constructor for the Graph class
   * @param type: either directed or undirected. Default = undirected
   */
    Graph(const std::string &type = "undirected");

    /**
     * @brief Insert an edge between two nodes in the graph
     * @param u: first node
     * @param v: second node
     * @param key: weight of u-v edge
     */
    void insert_edge(const unsigned int &u, const unsigned int &v,
                     const double &key);
    /**
     * @brief Removes a node from the graph
     * @param u: node to be removed. If u doesn't exist it fails silently
     */
    void remove_node(const unsigned int &u);
    /**
     * @brief Returns number of nodes in the graph
     * @return size_t: the number of nodes in the graph
     */
    size_t n_nodes() const;
    /**
     * @brief Returns number of edges in the graph
     * @return size_t: the number of edges in the graph
     */
    size_t n_edges() const;
    /**
     * @brief Returns shortest paths using floyd warshall's algorithm
     * @return std::vector<double>: a flattened 1-D array that holds the shortest paths between all the nodes in the graph
     */
    std::vector<double> floyd_warshall_all_shortest_paths();
    /**
     * @brief Returns shortest paths using bellman ford's algorithm
     * @param s: starting node
     * @return std::vector<double>: a flattened 1-d array that holds the shortest paths between all the nodes in the graph
     */
    std::vector<double> bellman_ford(const unsigned int &s);
    /**
     * @brief Returns shortest paths using dijkstra's algorithm
     * @param s: starting node
     *
     * @return std::vector<double>: shortest paths distances from s
     */
    std::vector<double> dijkstra_from_single(const unsigned int &s);
    /**
     * @brief Returns all the shortest paths in the graph using dijkstra's algorithm
     *
     * @return std::vector<double>: a flattened 1-d array that holds the shortest paths between all the nodes in the graph
     */
    std::vector<std::vector<double>> dijkstra_all_shortest_paths();
};
} // namespace clusterxx

#include "graph_impl.hpp"

#endif
