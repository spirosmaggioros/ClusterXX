#include <clusterxx.hpp>

int main() {
    clusterxx::csv_parser parser = clusterxx::csv_parser("../data/circles_05.csv", {"X", "Y"});
    arma::mat data = parser.data();
}
