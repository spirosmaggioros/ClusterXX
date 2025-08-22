sources = [
    'clusterxx/methods/tsne/tsne.cc',
]

libclusterxx = static_library(
    'clusterxx',
    sources: sources,
    include_directories: include_directories('.'),
    install: true,
)

install_headers('clusterxx.hpp', subdir: '')
install_subdir('.', install_dir: 'include', exclude_files: ['meson.build', '*.cc'])

