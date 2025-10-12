#include <clusterxx.hpp>
#include <catch2/catch_test_macros.hpp>


TEST_CASE("Testing reading a csv file", "[reading]") {
    std::vector<std::string> selected_cols = {"Year", "Mileage_KM", "Price_USD"};
    clusterxx::csv_parser parser = clusterxx::csv_parser("tests/reading/reading_data/bmw_sales_data.csv", selected_cols);
    arma::mat data;
    REQUIRE_NOTHROW(data = parser.data());
}
