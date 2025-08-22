#include "tsne.hpp"
#include "json.hpp"
#include <fstream>
#include <iostream>

using json = nlohmann::json;

int main() {
    clusterxx::TSNE tsne = clusterxx::TSNE(2, 3, 100, 1000);
    

    std::vector<std::vector<double> > X = {{0, 0, 0}, {-1, -1, -1}, {140, 140, 140}, {150, 150, 150}};
    std::vector<std::vector<double> > TSNE_features = tsne.fit_transform(X);
    
    json data;
    data["data"] = TSNE_features;
    std::ofstream file("features.json");
    file << data.dump(4);
}
