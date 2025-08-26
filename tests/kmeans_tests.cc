#include <catch2/catch_test_macros.hpp>
#include <clusterxx.hpp>
#include <random>
#include <vector>

TEST_CASE("Testing KMeans fit method with 2D data", "[kmeans]") {
    arma::mat X(2000, 2);

    for (int i = 0; i < 1000; i++) {
        std::vector<double> _curr;
        _curr.push_back(rand() % 3);
        _curr.push_back(rand() % 3);
        X.row(i) = arma::vec(_curr).t();
    }

    for (int i = 0; i < 1000; i++) {
        std::vector<double> _curr;
        _curr.push_back(25 + rand() % (30 - 25 + 1));
        _curr.push_back(25 + rand() % (30 - 25 + 1));
        X.row(1000 + i) = arma::vec(_curr).t();
    }

    clusterxx::KMeans kmeans = clusterxx::KMeans(2);
    REQUIRE_NOTHROW(kmeans.fit(X));

    std::vector<int> labels = kmeans.get_labels();
    REQUIRE(labels.size() == 2000);

    arma::mat to_predict = {{0, 0}, {50, 50}};
    labels = kmeans.predict(to_predict);

    if (labels[0] == 0) {
        REQUIRE(labels[1] == 1);
    } else {
        REQUIRE(labels[1] == 0);
    }
}

TEST_CASE("Testing KMeans with a different metric", "[kmeans]") {
    // Note that you should use a clusterxx::pairwise_distances metric for clustering

    arma::mat X = {{0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}, {10.0, 10.0, 10.0}, {11.0, 11.0, 11.0}};
    clusterxx::KMeans<clusterxx::pairwise_distances::manhattan_distances> kmeans = clusterxx::KMeans<clusterxx::pairwise_distances::manhattan_distances>(2);
    REQUIRE_NOTHROW(kmeans.fit(X));

    std::vector<int> labels = kmeans.get_labels();
    REQUIRE(labels.size() == 4);
    REQUIRE(labels[0] != labels[2]);
    REQUIRE(labels[1] != labels[3]);
}

TEST_CASE("Testing with random mode", "[kmeans]") {
    arma::mat X = {{0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}, {10.0, 10.0, 10.0}, {11.0, 11.0, 11.0}};
    clusterxx::KMeans kmeans = clusterxx::KMeans(/* n_clusters */ 2, /* max_iter */ 300, /* init */ "random");
    REQUIRE_NOTHROW(kmeans.fit(X));
    std::vector<int> labels = kmeans.get_labels();
    REQUIRE(labels[0] != labels[3]);
}

TEST_CASE("Testing with static random state", "[kmeans]") {
    arma::mat X = {{0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}, {10.0, 10.0, 10.0}, {11.0, 11.0, 11.0}};
    clusterxx::KMeans kmeans = clusterxx::KMeans(/* n_clusters */ 2, /* max_iter */ 300, /* init */ "random", /* random state */ 42);
    REQUIRE_NOTHROW(kmeans.fit(X));
    std::vector<int> labels = kmeans.get_labels();
    REQUIRE(labels[0] != labels[3]);
}

TEST_CASE("Testing high dimensional data", "[kmeans]") {
    arma::mat X(2000, 15);
    for (int i = 0; i < 1000; i++) {
        std::vector<double> _curr;
        for (int j = 0; j < 15; j++) {
            _curr.push_back(rand() % 3);
        }
        X.row(i) = arma::vec(_curr).t();
    }

    for (int i = 0; i < 1000; i++) {
        std::vector<double> _curr;
        for (int j = 0; j < 15; j++) {
            _curr.push_back(25 + rand() % (30 - 25 + 1));
        }
        X.row(1000 + i) = arma::vec(_curr).t();
    }

    clusterxx::KMeans kmeans = clusterxx::KMeans(2);
    std::vector<int> labels = kmeans.fit_predict(X);
    arma::mat centroids = kmeans.get_centroids();
    REQUIRE(centroids.n_rows == 2);
    REQUIRE(centroids.n_cols == 15);
    REQUIRE(labels.size() == 2000);

    arma::mat to_predict(1, 15);
    std::vector<double> feat;
    for (int i = 0; i < 15; i++) {
        feat.push_back(20 + rand() % (35 - 20 + 1));
    }
    to_predict.row(0) = arma::vec(feat).t();

    // Note that you can predict when you've performed fit_predict().
    // We don't destroy the state when you run fit_predict, we could've worked
    // just with fit(), but we wanted to make the API just like scikit's.
    std::vector<int> predicted_labels = kmeans.predict(to_predict);
    REQUIRE(predicted_labels[0] == labels[1001]);
}

TEST_CASE("Testing high dimensional data with >2 clusters", "[kmeans]") {
    arma::mat X(20000, 30);
    for (int i = 0; i < 5000; i++) {
        std::vector<double> _curr;
        for (int j = 0; j < 30; j++) {
            _curr.push_back(rand() % 3);
        }
        X.row(i) = arma::vec(_curr).t();
    }

    for (int i = 0; i < 10000; i++) {
        std::vector<double> _curr;
        for (int j = 0; j < 30; j++) {
            _curr.push_back(35 + rand() % (35 - 25 - 1));
        }
        X.row(5000 + i) = arma::vec(_curr).t();
    }

    for (int i = 0; i < 5000; i++) {
        std::vector<double> _curr;
        for (int j = 0; j < 30; j++) {
            _curr.push_back(120 + rand() % (120 + 100 - 1));
        }
        X.row(15000 + i) = arma::vec(_curr).t();
    }

    clusterxx::KMeans kmeans = clusterxx::KMeans(3);
    std::vector<int> labels = kmeans.fit_predict(X);
    REQUIRE(labels[0] != labels[10001]);
    REQUIRE(labels[10001] != labels[15001]);
}
