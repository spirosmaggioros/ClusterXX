#include <catch2/catch_test_macros.hpp>
#include <clusterxx.hpp>
#include <vector>

TEST_CASE("Testing graph shortest path with floyd warshall", "[Graph]") {
    clusterxx::Graph g;
    g.insert_edge(0, 1, 6);
    g.insert_edge(0, 2, 1);
    g.insert_edge(1, 2, 2);
    g.insert_edge(1, 3, 2);
    g.insert_edge(1, 4, 5);
    g.insert_edge(2, 3, 1);
    g.insert_edge(3, 4, 5);
    std::vector<double> dists = g.floyd_warshall();

    REQUIRE(dists[4] == 7);
    REQUIRE(dists[5 * 2 + 4] == 6);
}

