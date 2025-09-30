#ifndef CLUSTERXX_THIRD_PARTY_CSV_PARSER_IMPL_HPP
#define CLUSTERXX_THIRD_PARTY_CSV_PARSER_IMPL_HPP

#include "clusterxx/third_party/csv.hpp"
#include "csv_parser.hpp"
#include <iostream>

clusterxx::csv_parser::csv_parser(const std::string &csv_file) {
    csv::CSVReader reader(csv_file);
    std::vector<std::string> col_names = csv::get_col_names(csv_file);
    const size_t n_rows = reader.n_rows();
    const size_t n_cols = col_names.size();
    if (n_rows == 0 || n_cols == 0) {
        std::cout << "[WARNING] Parsing empty csv file" << '\n';
    }

    __data.resize(n_rows, n_cols);

    size_t i = 0;
    for (const auto &row : reader) {
        for (size_t j = 0; j < n_cols; j++) {
            __data(i, j) = row[col_names[j]].get<double>();
        }
        i++;
    }
}

arma::mat clusterxx::csv_parser::data() const { return __data; }

#endif
