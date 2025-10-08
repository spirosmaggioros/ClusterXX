#ifndef CLUSTERXX_THIRD_PARTY_CSV_PARSER_HPP
#define CLUSTERXX_THIRD_PARTY_CSV_PARSER_HPP

#include <armadillo>
#include <string>

namespace clusterxx {
class csv_parser {
  private:
    arma::mat __data;

  public:
    csv_parser(const std::string &csv_file);
    ~csv_parser() {}

    arma::mat data() const;
};
} // namespace clusterxx

#include "csv_parser_impl.hpp"

#endif
