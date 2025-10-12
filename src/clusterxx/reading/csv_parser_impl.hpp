#ifndef CLUSTERXX_THIRD_PARTY_CSV_PARSER_IMPL_HPP
#define CLUSTERXX_THIRD_PARTY_CSV_PARSER_IMPL_HPP

#include "clusterxx/third_party/csv.hpp"
#include "csv_parser.hpp"

#include <filesystem>

clusterxx::csv_parser::csv_parser(const std::string &csv_file, const std::vector<std::string> &selected_cols) {
    std::string abs_csv_file = std::filesystem::absolute(csv_file);
    csv::CSVReader reader(abs_csv_file);
    std::vector<std::string> col_names;
    if (selected_cols.empty()) {
        col_names = csv::get_col_names(abs_csv_file);
    } else {
        col_names = selected_cols;
    }
    const size_t n_cols = col_names.size();

    __data.resize(0, n_cols);

    size_t i = 0;
    for (const auto &row : reader) {
        __data.insert_rows(__data.n_rows, 1);
        for (size_t j = 0; j < n_cols; j++) {
            __data(i, j) = row[col_names[j]].get<double>();
        }
        i++;
    }
}

arma::mat clusterxx::csv_parser::data() const { return __data; }

#endif
