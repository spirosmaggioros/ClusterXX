#include <clusterxx.hpp>

int main() {
    clusterxx::csv_parser parser = clusterxx::csv_parser("../data/mnist_test.csv");
    arma::mat data = parser.data();

    clusterxx::isomap<> isomap = clusterxx::isomap<>(
        5, /* n_neighbors */
        0.0, /* radius(set to 0.0 if n_neighbors is != 0) */
        2, /* n_components */
        "auto" /* path_method */
    );
    auto latent_features = isomap.fit_transform(data);

    clusterxx::Plot plot;
    plot.plot2d(isomap);
}
