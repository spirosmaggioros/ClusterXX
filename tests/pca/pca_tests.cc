#include <armadillo>
#include <clusterxx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("Testing default PCA", "[PCA]") {
    arma::mat features = {{-1, -1}, {-2, -1}, {-3, -2}, {1, 1}, {2, 1}, {3, 2}};

    clusterxx::PCA pca = clusterxx::PCA(2);
    arma::mat pca_features = pca.fit_transform(features);
    arma::vec explained_variance_ = pca.get_explained_variance();
    arma::vec check = {7.93954312, 0.06045688};
    for (size_t i = 0; i < 2; i++) {
        REQUIRE_THAT(explained_variance_(i), Catch::Matchers::WithinAbs(check(i), 1e-5));
    }

    arma::mat check_2 = {{-1.38340578, -0.2935787 },
       {-2.22189802,  0.25133484},
       {-3.6053038 , -0.04224385},
       { 1.38340578,  0.2935787 },
       { 2.22189802, -0.25133484},
       { 3.6053038 ,  0.04224385}};
    for (size_t i = 0; i < features.n_rows; i++) {
        for (size_t j = 0; j < features.n_cols; j++) {
            REQUIRE_THAT(std::abs(pca_features(i, j)), Catch::Matchers::WithinAbs(std::abs(check_2(i, j)), 1e-5));
        }
    }
}
