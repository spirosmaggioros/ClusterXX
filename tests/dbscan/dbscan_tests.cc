#include <catch2/catch_test_macros.hpp>
#include <clusterxx.hpp>
#include <random>
#include <vector>

TEST_CASE("Testing DBSCAN fit method with 2d data", "[DBSCAN]") {
    arma::mat X = {{1, 2}, {2, 2}, {2, 3}, {8, 7}, {8, 8}, {25, 80}};
    clusterxx::DBSCAN clustering = clusterxx::DBSCAN(3, 2);
    std::vector<int> labels = clustering.fit_predict(X);
    std::vector<int> check = {0, 0, 0, 1, 1, -1};
    REQUIRE(labels == check);
}

TEST_CASE("Testing DBSCAN with different leaf size", "[DBSCAN]") {
    arma::mat X = {{1, 1, 1}, {0.5, 0.5, 0.5}, {0.2, 0.2, 0.2},
                   {10.5, 10.3, 10.2}, {10.4, 10.4, 10.4}};
    clusterxx::DBSCAN clustering = clusterxx::DBSCAN(1.0, 2, 2);
    std::vector<int> labels = clustering.fit_predict(X);
    std::vector<int> check = {0, 0, 0, 1, 1};
    REQUIRE(labels == check);
}
