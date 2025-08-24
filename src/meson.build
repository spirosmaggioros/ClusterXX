sources = []

clusterxx_inc = include_directories('.')

libclusterxx = declare_dependency(
    include_directories: clusterxx_inc,
)

install_headers('clusterxx.hpp', subdir: '')
install_subdir('.', install_dir: 'include', exclude_files: ['meson.build', '*.cc'])

