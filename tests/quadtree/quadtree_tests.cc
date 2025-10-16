#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <clusterxx.hpp>
#include <armadillo>


TEST_CASE("Testing quadtree construction", "[quadtree]") {
    arma::mat X(100, 3);

    for (int i = 0; i < 100; i++) {
        std::vector<double> curr;
        for (int j = 0; j < 3; j++) {
            curr.push_back(rand() % 1000);
        }
        X.row(i) = arma::rowvec(curr);
    }

    clusterxx::quadtree<3> quadtree = clusterxx::quadtree<3>(X);
}
