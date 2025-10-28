#include <clusterxx.hpp>

int main() {
    clusterxx::csv_parser parser = clusterxx::csv_parser("../data/mnist_test.csv");
    arma::mat data = parser.data();
    
    clusterxx::PCA pca = clusterxx::PCA(30 /* n_components */);
    auto latent_features = pca.fit_transform(data);
}

