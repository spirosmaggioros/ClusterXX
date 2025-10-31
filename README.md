# ClusterXX
ClusterXX is a C++ library that includes **clustering, manifold and decomposition** algorithms as well as the required data structures for them to be fast. Everything is implemented from scratch with armadillo being the only external library. The API follows **sklearn's** API so that you don't have to read all of our documentation.

> [!Note]
> This library is semi-educational, meaning that, though i implemented any data structures needed to make the algorithms faster, they are not as fast as sklearn's, and i don't know if i'll ever make them as fast.

## Example:
```cpp
#include <clusterxx.hpp>

int main() {
    clusterxx::csv_parser parser = clusterxx::csv_parser("dataset.csv");
    auto X = parser.data();
    clusterxx::DBSCAN dbscan = clusterxx::DBSCAN(
        0.5, /* eps */
        5, /* num_samples */
        30, /* leaf_size */
    );
    std::vector<int> labels = dbscan.fit_predict(X);

    clusterxx::PCA pca = clusterxx::PCA(30 /* n_components */);
    auto latent_features = pca.fit_transform(X);

    // We also support simple plotting
    clusterxx::Plot plot;
    plot.plot2d(dbscan);
}
```

You can see more examples at [examples](/examples)

## Currently implemented methods and data structures:
- [X] DBSCAN
- [X] Isomap
- [X] KMeans
- [X] Mini Batch KMeans
- [X] PCA
- [X] t-SNE(exact method only)
- [X] k-d tree
- [X] Vantage point tree
- [X] Quadtree
- [X] Simple Graph for shortest paths(Isomap)

## Installation:
First you need to install armadillo

At root directory, do:
**Linux/MacOS**
```bash
meson install -C build
```

**Windows**
Need help here as i can't test it!

To run the unit tests, you can do:
```bash
meson test -C build
```

## Contributions:
Contributions are open, you can contribute by solving open issues or by submitting a PR with an implementation/addition.
For any information or question contact spiros at **spirosmag@ieee.org**
