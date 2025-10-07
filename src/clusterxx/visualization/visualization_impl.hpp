#ifndef CLUSTERXX_VISUALIZATION_IMPL_HPP
#define CLUSTERXX_VISUALIZATION_IMPL_HPP

#include "clusterxx/writing/write_json.hpp"
#include "visualization.hpp"

// just for now
#include <iostream>

void clusterxx::Plot::scatter_plot() {
    return;
}

void clusterxx::Plot::plot(const cluster_method &m) {
    std::cout << m.get_in_features() << '\n';
    std::string tmp_file = 
        clusterxx::save_to_tmp_json_clustering(m.get_in_features(), m.get_labels());
    std::string cmd = "python3 -m clusterxx_visualization clustering --filename " + tmp_file;
    system(cmd.c_str());
}

#endif
