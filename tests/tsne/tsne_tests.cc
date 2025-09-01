#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <clusterxx.hpp>
#include <chrono>
#include <random>
#include <iostream>
#include <armadillo>

TEST_CASE("temp test", "[t-SNE]") {
    REQUIRE(true == true);
}


// TOO SLOW
// TEST_CASE("Testing t-SNE with low dimensional data", "[t-SNE]") {
//     arma::mat X(5000, 3);
// 
//     for (int i = 0; i < 5000; i++) {
//         std::vector<double> curr;
//         for (int j = 0; j < 3; j++) {
//             curr.push_back(rand() % 5);
//         }
//         X.row(i) = arma::vec(curr).t();
//     }
// 
//     clusterxx::TSNE tsne = clusterxx::TSNE(
//             /* n_components */2,
//             /* perplexity */ 3, // have to, perplexity <= X.n_cols
//             /* learning_rate */ 100,
//             /* early_exaggeration */ 4.0,
//             /* max iter */ 1000
//     );
//     REQUIRE_NOTHROW(tsne.fit(X));
// }
