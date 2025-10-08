project('clusterxx', 'cpp',
  version: '0.1',
  default_options: [
    'cpp_std=c++23',
    'optimization=3',
    'b_lto=true',
  ])

subdir('src')
subdir('tests')

pkg = import('pkgconfig')

pkg.generate(
  name : 'clusterxx',
  description : 'Clustering made easy in modern C++',
  filebase : 'clusterxx',
  subdirs : '.',   # headers are installed into /usr/local/include
  libraries : libclusterxx,
  version : meson.project_version()
  )
