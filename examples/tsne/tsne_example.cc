#include <clusterxx.hpp>

int main() {
    clusterxx::csv_parser parser = clusterxx::csv_parser("../data/mnist_test.csv");
    arma::mat data = parser.data();

    clusterxx::TSNE<> tsne = clusterxx::TSNE<>(
        2, /* n_components */
        30.0, /* complexity(use between 30 - 50) */
        200.0, /* learning_rate */
        12.0, /* early_exaggeration */
        1000, /* max_iter */
        1e-7, /* min_grad_norm */
        300 /* n_iter_without_progress */
    );
    auto latent_features = tsne.fit_transform(data);

    clusterxx::Plot plot;
    plot.plot2d(tsne);
}
