sources = []
clusterxx_inc = include_directories('.')

armadillo_dep = dependency('armadillo', required: false)

if not armadillo_dep.found()
  armadillo_inc = include_directories('../extern/armadillo/include')
  armadillo_dep = declare_dependency(
    include_directories: armadillo_inc
  )
endif

libclusterxx = declare_dependency(
    include_directories: clusterxx_inc,
    dependencies: [armadillo_dep],
    link_args: ['-framework', 'Accelerate'],
)

install_headers('clusterxx.hpp', subdir: '')
install_subdir('.', install_dir: 'include', exclude_files: ['meson.build', '*.cc'])

