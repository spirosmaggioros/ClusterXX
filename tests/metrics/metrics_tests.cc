#include "clusterxx/metrics/metrics.hpp"
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

TEST_CASE("Testing silhouette score", "[metrics]") {
    arma::mat X = {{-7.72642091, -8.39495682},
       { 5.45339605,  0.74230537},
       {-2.97867201,  9.55684617},
       { 6.04267315,  0.57131862},
       {-6.52183983, -6.31932507},
       { 3.64934251,  1.40687195},
       {-2.17793419,  9.98983126},
       { 4.42020695,  2.33028226},
       { 4.73695639,  2.94181467},
       {-3.6601912 ,  9.38998415},
       {-3.05358035,  9.12520872},
       {-6.65216726, -5.57296684},
       {-6.35768563, -6.58312492},
       {-3.6155326 ,  7.8180795 },
       {-1.77073104,  9.18565441},
       {-7.95051969, -6.39763718},
       {-6.60293639, -6.05292634},
       {-2.58120774, 10.01781903},
       {-7.76348463, -6.72638449},
       {-6.40638957, -6.95293851},
       {-2.97261532,  8.54855637},
       {-6.9567289 , -6.53895762},
       {-7.32614214, -6.0237108 },
       {-2.14780202, 10.55232269},
       {-2.54502366, 10.57892978},
       {-2.96983639, 10.07140835},
       { 3.22450809,  1.55252436},
       {-6.25395984, -7.73726715},
       {-7.85430886, -6.09302499},
       {-8.1165779 , -8.20056621},
       {-7.55965191, -6.6478559 },
       { 4.93599911,  2.23422496},
       { 4.44751787,  2.27471703},
       {-5.72103161, -7.70079191},
       {-0.92998481,  9.78172086},
       {-3.10983631,  8.72259238},
       {-2.44166942,  7.58953794},
       {-2.18511365,  8.62920385},
       { 5.55528095,  2.30192079},
       { 4.73163961, -0.01439923},
       {-8.25729656, -7.81793463},
       {-2.98837186,  8.82862715},
       { 4.60516707,  0.80449165},
       {-3.83738367,  9.21114736},
       {-2.62484591,  8.71318243},
       { 3.57757512,  2.44676211},
       {-8.48711043, -6.69547573},
       {-6.70644627, -6.49479221},
       {-6.8666253 , -5.42657552},
       { 3.83138523,  1.47141264},
       { 2.02013373,  2.79507219},
       { 4.64499229,  1.73858255},
       {-1.6966718 , 10.37052616},
       {-6.6197444 , -6.09828672},
       {-6.05756703, -4.98331661},
       {-7.10308998, -6.1661091 },
       {-3.52202874,  9.32853346},
       {-2.26723535,  7.10100588},
       { 6.11777288,  1.45489947},
       {-4.23411546,  8.4519986 },
       {-6.58655472, -7.59446101},
       { 3.93782574,  1.64550754},
       {-7.12501531, -7.63384576},
       { 2.72110762,  1.94665581},
       {-7.14428402, -4.15994043},
       {-6.66553345, -8.12584837},
       { 4.70010905,  4.4364118 },
       {-7.76914162, -7.69591988},
       { 4.11011863,  2.48643712},
       { 4.89742923,  1.89872377},
       { 4.29716432,  1.17089241},
       {-6.62913434, -6.53366138},
       {-8.07093069, -6.22355598},
       {-2.16557933,  7.25124597},
       { 4.7395302 ,  1.46969403},
       {-5.91625106, -6.46732867},
       { 5.43091078,  1.06378223},
       {-6.82141847, -8.02307989},
       { 6.52606474,  2.1477475 },
       { 3.08921541,  2.04173266},
       {-2.1475616 ,  8.36916637},
       { 3.85662554,  1.65110817},
       {-1.68665271,  7.79344248},
       {-5.01385268, -6.40627667},
       {-2.52269485,  7.9565752 },
       {-2.30033403,  7.054616  },
       {-1.04354885,  8.78850983},
       { 3.7204546 ,  3.52310409},
       {-3.98771961,  8.29444192},
       { 4.24777068,  0.50965474},
       { 4.7269259 ,  1.67416233},
       { 5.78270165,  2.72510272},
       {-3.4172217 ,  7.60198243},
       { 5.22673593,  4.16362531},
       {-3.11090424, 10.86656431},
       {-3.18611962,  9.62596242},
       {-1.4781981 ,  9.94556625},
       { 4.47859312,  2.37722054},
       {-5.79657595, -5.82630754},
       {-3.34841515,  8.70507375}};

    clusterxx::KMeans kmeans = clusterxx::KMeans<>(2);
    clusterxx::clustering::silhouette_score<> score;
    std::vector<int> labels = kmeans.fit_predict(X);
    REQUIRE_NOTHROW(score(X, labels));

    X = {
        {0, 0, 0},
        {1, 1, 1},
        {2, 2, 2},
        {10, 10, 10},
        {12, 12, 12},
        {11.5, 11.5, 11.5}
    };
    labels = kmeans.fit_predict(X);
    REQUIRE(score(X, labels) > 0.5);
}

