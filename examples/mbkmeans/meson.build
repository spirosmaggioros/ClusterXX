project('examples', 'cpp', default_options: [
  'cpp_std=c++20',
  'optimization=3',
  'b_lto=true',
  ])

add_project_arguments(
  '-DARMA_USE_BLAS',
  '-DARMA_USE_LAPACK',
  language: 'cpp'
  )

clusterxx_dep = dependency('clusterxx', required: true)
armadillo_dep = dependency('armadillo', required: true)

executable('main',
  'mbkmeans_example.cc',
  dependencies: [clusterxx_dep, armadillo_dep]
  )
