#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <clusterxx.hpp>

#include <iostream>

TEST_CASE("Testing euclidean distance", "[metrics]") {
    clusterxx::metrics::euclidean_distance dist;

    std::vector<double> a = {1, 2, 3};
    std::vector<double> b = {2.5, 3.5, 4.5};

    REQUIRE(dist(a, b) == 2.59807621135331601);

    a = {0.0, 0.0, 0.0, 0.0, 0.0};
    b = {0.0, 0.0, 0.0, 0.0, 0.0};

    REQUIRE(dist(a, b) == 0.0);

    a = {2.5};
    b = {5.0};

    REQUIRE(dist(a, b) == 2.5);
}

TEST_CASE("Testing euclidean pairwise distances", "[metrics]") {
    clusterxx::pairwise_distances::euclidean_distances dist;

    std::vector<std::vector<double>> X = {{1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}};
    std::vector<std::vector<double>> pairwise_dists = dist(X);
    std::vector<std::vector<double>> check = {{0.0, 1.41421, 2.82843}, {1.41421, 0, 1.41421}, {2.82843, 1.41421, 0.0}};

    for (int i = 0; i < check.size(); i++) {
        for (int j = 0; j < check[0].size(); j++) {
            REQUIRE_THAT(pairwise_dists[i][j], Catch::Matchers::WithinAbs(check[i][j], 1e-5));
        }
    }
}

TEST_CASE("Testing manhattan distance", "[metrics]") {
    REQUIRE(true == true);
}

