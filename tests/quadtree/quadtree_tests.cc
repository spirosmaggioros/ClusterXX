#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <clusterxx.hpp>
#include <armadillo>
#include <ranges>


TEST_CASE("Testing quadtree construction", "[quadtree]") {
    arma::mat X(100, 2);

    for (int i = 0; i < 100; i++) {
        std::vector<double> curr;
        for (int j = 0; j < 2; j++) {
            curr.push_back(rand() % 1000);
        }
        X.row(i) = arma::rowvec(curr);
    }

    clusterxx::quadtree quadtree = clusterxx::quadtree(X);
}

TEST_CASE("Testing quadtree range query", "[quadtree]") {
    arma::mat X {
        {-5, 5},
        {-2, 2},
        {3, -3},
        {3.5, -3.5},
        {4, -4}
    };

    clusterxx::quadtree quadtree = clusterxx::quadtree(X);

    arma::vec point(2);
    point(0) = 0;
    point(1) = 0;
    std::vector<size_t> pts_in_range = quadtree.range_query(point, 4.5);
    REQUIRE(std::ranges::find(pts_in_range, 1) != pts_in_range.end());
    REQUIRE(std::ranges::find(pts_in_range, 2) != pts_in_range.end());
    REQUIRE(std::ranges::find(pts_in_range, 3) != pts_in_range.end());
    REQUIRE(std::ranges::find(pts_in_range, 4) != pts_in_range.end());
}
