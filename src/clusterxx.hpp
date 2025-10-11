#ifndef CLUSTERXX_HPP
#define CLUSTERXX_HPP

// metrics
#include "clusterxx/metrics/metrics.hpp"

// data structures
#include "clusterxx/data_structures/kd_tree/kd_tree.hpp"
#include "clusterxx/data_structures/vp_tree/vp_tree.hpp"
#include "clusterxx/data_structures/graph/graph.hpp"

// cluster
#include "clusterxx/methods/kmeans_plus_plus/kmeans_plus_plus.hpp"
#include "clusterxx/methods/kmeans/kmeans.hpp"
#include "clusterxx/methods/mbkmeans/mbkmeans.hpp"
#include "clusterxx/methods/dbscan/dbscan.hpp"

// manifold
#include "clusterxx/methods/tsne/tsne.hpp"
#include "clusterxx/methods/isomap/isomap.hpp"

// decomposition
#include "clusterxx/methods/pca/pca.hpp"

// csv parsing
#include "clusterxx/reading/csv_parser.hpp"

// writing
#include "clusterxx/writing/write_json.hpp"

// visualization
#include "clusterxx/visualization/visualization.hpp"
#endif
