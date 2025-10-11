#include <catch2/catch_test_macros.hpp>
#include <clusterxx.hpp>
#include <armadillo>

TEST_CASE("Testing vp_tree construction", "[vp_tree]") {
    arma::mat X(100, 3);

    for (int i = 0; i < 100; i++) {
        std::vector<double> curr;
        for (int j = 0; j < 3; j++) {
            curr.push_back(rand() % 1000);
        }
        X.row(i) = arma::rowvec(curr);
    }

    clusterxx::vp_tree<> vp_tree = clusterxx::vp_tree<>(X);

    REQUIRE(vp_tree.depth() < 10);
}


TEST_CASE("Testing vp_tree searching", "[vp_tree]") {
    arma::mat X {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}, {5, 5, 5}, {6, 6, 6},
        {7, 7, 7}, {8, 8, 8}, {2.5, 2.5, 2.5}, {4.5, 4.5, 4.5}};
    clusterxx::vp_tree<> vp_tree = clusterxx::vp_tree<>(X);

    arma::vec p = {4, 4, 4};
    auto [inds, dists] = vp_tree.query(p, 2);
    std::vector<int> inds_check = {9, 3};
    REQUIRE(inds == inds_check);
}
