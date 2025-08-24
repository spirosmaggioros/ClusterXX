#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <clusterxx.hpp>

TEST_CASE("Testing euclidean distance", "[metrics]") {
    clusterxx::metrics::euclidean_distance dist;

    std::vector<double> a = {1, 2, 3};
    std::vector<double> b = {2.5, 3.5, 4.5};

    REQUIRE(dist(a, b) == 2.59807621135331601);
}


