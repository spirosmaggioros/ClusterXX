#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <clusterxx.hpp>
#include <armadillo>


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
    std::cout << quadtree.depth() << '\n';
}

// TEST_CASE("Testing quadtree range query", "[quadtree]") {
//     arma::mat X {
//         {-5, 5},
//         {-2, 2},
//         {-3, 3},
//         {-3.5, 3.5},
//         {-4, -4}
//     };
// 
//     clusterxx::quadtree quadtree = clusterxx::quadtree(X);
// 
//     arma::vec center = {0, 0};
//     std::vector<size_t> pts_in_range = quadtree.range_query(center, 4.5);
// 
//     for (auto &x: pts_in_range) {
//         std::cout << x << ' ';
//     }
//     std::cout << '\n';
// }
