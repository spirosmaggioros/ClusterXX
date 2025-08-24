#include <catch2/catch_test_macros.hpp>
#include <clusterxx.hpp>
#include <random>
#include <vector>


TEST_CASE("Testing KMeans fit method with 2D data", "[kmeans]") {
    std::random_device dev;
    std::mt19937 rng(dev());

    std::vector<std::vector<double>> X;

    for (int i = 0; i<1000; i++) {
        std::uniform_int_distribution<std::mt19937::result_type> dist(-3, 3);
        std::vector<double> _curr;
        _curr.push_back(dist(rng));
        _curr.push_back(dist(rng));
        X.push_back(_curr);
    }

    for (int i = 0; i<1000; i++) {
        std::uniform_int_distribution<std::mt19937::result_type> dist(30, 35);
        std::vector<double> _curr;
        _curr.push_back(dist(rng));
        _curr.push_back(dist(rng));
        X.push_back(_curr);
    }

    clusterxx::KMeans kmeans = clusterxx::KMeans(2);
    REQUIRE_NOTHROW(kmeans.fit(X));

    std::vector<int> labels = kmeans.get_labels();
    REQUIRE(labels.size() == 2000);
}
