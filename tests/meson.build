catch2_dep = dependency('catch2-with-main', required: true)

clusterxx_dep = declare_dependency(
    include_directories: include_directories('../src')
)

tests = {
    'metrics_tests': 'metrics_tests.cc',
    'kmeans_tests': 'kmeans_tests.cc',
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
