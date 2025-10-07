#ifndef CLUSTERXX_VISUALIZATION_IMPL_HPP
#define CLUSTERXX_VISUALIZATION_IMPL_HPP

#include "clusterxx/writing/write_json.hpp"
#include "visualization.hpp"


void clusterxx::Plot::plot2d(const cluster_method &m, const std::string &title,
        const std::string &xlabel, const std::string &ylabel) {
    std::string tmp_file = clusterxx::save_to_tmp_json(
        m.get_in_features(), m.get_labels());
    std::string cmd =
        "python3 -m clusterxx_visualization clustering --filename " + tmp_file
        + " --title " + title + " --xlabel " + xlabel + " --ylabel " + ylabel;
    system(cmd.c_str());
}

void clusterxx::Plot::plot3d(const cluster_method &m, const std::string &title,
        const std::string &xlabel, const std::string &ylabel,
        const std::string &zlabel) {
    std::string tmp_file = clusterxx::save_to_tmp_json(
            m.get_in_features(), m.get_labels());
    std::string cmd =
        "python3 -m clusterxx_visualization clustering --filename " + tmp_file
        + " --title " + title + " --xlabel " + xlabel + " --ylabel " + ylabel
        + " --zlabel " + zlabel;
    system(cmd.c_str());
}


template <typename T>
void clusterxx::Plot::plot2d(const T &m, const std::string &title,
        const std::string &xlabel, const std::string &ylabel,
        const std::vector<int> &labels) {
    std::string tmp_file = clusterxx::save_to_tmp_json(
        m.get_in_features(), labels);
    std::string cmd =
        "python3 -m clusterxx_visualization scatterplot --filename " + tmp_file
        + " --title " + title + " --xlabel " + xlabel + " --ylabel " + ylabel;
    if (!labels.empty()) {
        cmd += " --labels";
    }
    system(cmd.c_str());
}

template <typename T>
void clusterxx::Plot::plot3d(const T &m, const std::string &title,
        const std::string &xlabel, const std::string &ylabel,
        const std::string &zlabel, const std::vector<int> &labels) {
    std::string tmp_file =
        clusterxx::save_to_tmp_json(m.get_in_features(), labels);
    std::string cmd =
        "python3 -m clusterxx_visualization scatterplot --filename " + tmp_file
        + " --title " + title + " --xlabel " + xlabel + " --ylabel " + ylabel
        + " --zlabel " + zlabel;

    if (!labels.empty()) {
        cmd += " --labels";
    }

    system(cmd.c_str());
}

#endif
