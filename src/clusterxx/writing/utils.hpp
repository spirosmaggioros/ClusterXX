#ifndef CLUSTERXX_WRITING_UTILS_HPP
#define CLUSTERXX_WRITING_UTILS_HPP

#include <armadillo>
#include <vector>

namespace clusterxx {
std::vector<std::vector<double>> mat2d_to_vec2d(const arma::mat &X) {
    std::vector<std::vector<double>> vec2d(X.n_rows,
                                           std::vector<double>(X.n_cols));
    for (size_t i = 0; i < X.n_rows; i++) {
        for (size_t j = 0; j < X.n_cols; j++) {
            vec2d[i][j] = X(i, j);
        }
    }

    return vec2d;
}
} // namespace clusterxx

#endif
