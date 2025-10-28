#include <clusterxx.hpp>

int main() {
    clusterxx::csv_parser parser = clusterxx::csv_parser("../data/circles_05.csv", {"X", "Y"});
    arma::mat data = parser.data();

    clusterxx::MiniBatchKMeans<clusterxx::pairwise_distances::euclidean_distances> kmeans =
        clusterxx::MiniBatchKMeans<clusterxx::pairwise_distances::euclidean_distances>(
            2, /* n_clusters */
            300, /* max_iter */
            1024, /* batch_size */
            "k-means++" /* init */
        ); // you can use checbyshev, manhattan or squared_euclidean if you wish
    std::vector<int> labels = kmeans.fit_predict(data);

    clusterxx::Plot plot;
    plot.plot2d(kmeans);
}
