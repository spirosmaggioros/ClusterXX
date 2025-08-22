project('clusterxx', 'cpp',
        version: '0.1',
        default_options: ['cpp_std=c++23'])

subdir('src')

pkg = import('pkgconfig')

pkg.generate(
  name : 'clusterxx',
  description : 'Clustering made easy in modern C++',
  filebase : 'clusterxx',
  subdirs : '.',   # headers are installed into /usr/local/include
  libraries : libclusterxx,
  version : meson.project_version()
)
