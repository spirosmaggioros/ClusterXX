#ifndef CLUSTERXX_WRITING_WRITE_JSON_HPP
#define CLUSTERXX_WRITING_WRITE_JSON_HPP

#include "clusterxx/third_party/json.hpp"
#include "clusterxx/writing/utils.hpp"

#include <armadillo>
#include <filesystem>
#include <fstream>
#include <vector>

namespace clusterxx {
void save_to_json_clustering(const arma::mat &X, const std::vector<int> &labels,
                             const std::string &filename) {
    nlohmann::json j;
    j["features"] = clusterxx::mat2d_to_vec2d(X);
    j["labels"] = labels;
    std::ofstream out_file(filename);
    out_file << j.dump(4);
}

std::string save_to_tmp_json_clustering(const arma::mat &X,
                                        const std::vector<int> &labels) {
    nlohmann::json j;
    j["features"] = clusterxx::mat2d_to_vec2d(X);
    j["labels"] = labels;
    std::filesystem::path tmp_filepath =
        std::filesystem::temp_directory_path() /
        std::filesystem::path("clusterxx-%%%%-%%%%.json");
    std::ofstream out_file(tmp_filepath);
    out_file << j.dump(4);
    return tmp_filepath.string();
}
} // namespace clusterxx

#endif
