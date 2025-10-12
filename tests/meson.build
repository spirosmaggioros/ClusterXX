catch2_dep = dependency('catch2-with-main', required: true)

clusterxx_dep = declare_dependency(
    include_directories: include_directories('../src')
)

tests = {
    'metrics_tests': 'metrics/metrics_tests.cc',
    'kmeans_tests': 'kmeans/kmeans_tests.cc',
    'dbscan_tests': 'dbscan/dbscan_tests.cc',
    'tsne_tests': 'tsne/tsne_tests.cc',
    'kd_tree_tests': 'kd_tree/kd_tree_tests.cc',
    'vp_tree_tests': 'vp_tree/vp_tree_tests.cc',
    'pca_tests': 'pca/pca_tests.cc',
    'graph_tests': 'graph/graph_tests.cc',
    'reading': 'reading/reading_tests.cc',
}

foreach test_name, test_source : tests

    test_exe = executable(
        test_name,
        test_source,
        dependencies: [clusterxx_dep, catch2_dep, libclusterxx],
        cpp_args: ['-std=c++23'],
        install: false
    )

    test(test_name, test_exe)
endforeach
