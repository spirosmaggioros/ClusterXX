#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <clusterxx.hpp>
#include <chrono>
#include <random>
#include <iostream>
#include <armadillo>

TEST_CASE("Testing euclidean distance", "[metrics]") {
    clusterxx::metrics::euclidean_distance dist;

    arma::vec a = {1, 2, 3};
    arma::vec b = {2.5, 3.5, 4.5};

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

    arma::mat X = {{1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}};
    arma::mat pairwise_dists = dist(X, {}); // linear run
    arma::mat check = {{0.0, 1.41421, 2.82843}, {1.41421, 0, 1.41421}, {2.82843, 1.41421, 0.0}};

    for (int i = 0; i < check.n_rows; i++) {
        for (int j = 0; j < check.n_cols; j++) {
            REQUIRE_THAT(pairwise_dists(i, j), Catch::Matchers::WithinAbs(check(i, j), 1e-5));
        }
    }
}

// TEST_CASE("Stress testing euclidean distance", "[metrics]") {
//     clusterxx::pairwise_distances::euclidean_distances dist;
// 
//     arma::mat X(5000, 30);
//     for (int i = 0; i < 5000; i++) {
//         std::vector<double> features;
//         for (int j = 0; j < 30; j++) {
//             features.push_back(rand() % 100);
//         }
//         X.row(i) = arma::vec(features).t();
//     }
//     
//     // auto t1 = std::chrono::high_resolution_clock::now();
//     arma::mat pairwise_dists_linear = dist(X, arma::mat());
//     // auto t2 = std::chrono::high_resolution_clock::now();
//     // std::chrono::duration<double, std::milli> ms_double = t2 - t1;
// 
//     arma::mat X_big(20000, 30);
//     for (int i = 0; i < 20000; i++) {
//         std::vector<double> features;
//         for (int j = 0; j < 30; j++) {
//             features.push_back(rand() % 100);
//         }
//         X_big.row(i) = arma::vec(features).t();
//     }
// 
//     auto t1 = std::chrono::high_resolution_clock::now();
//     arma::mat pairwise_dists_big = dist(X_big, arma::mat());
//     auto t2 = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> ms_double = t2 - t1;
//     std::cout << ms_double << '\n';
// }

TEST_CASE("Testing manhattan distance", "[metrics]") {
    REQUIRE(true == true);
}

