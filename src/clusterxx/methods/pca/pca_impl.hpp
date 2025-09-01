#ifndef CLUSTERXX_METHODS_PCA_IMPL_HPP
#define CLUSTERXX_METHODS_PCA_IMPL_HPP

#include "pca.hpp"

void clusterxx::PCA::__fit(const arma::mat &X) {
    arma::mat _X = X;
    arma::rowvec mean = arma::mean(X, 0);
    _X.each_row() -= mean;

    arma::mat Y = _X / std::sqrt(X.n_cols - 1);

    arma::mat U;
    arma::vec s;
    arma::mat V;

    arma::svd(U, s, V, Y);

    V = V.cols(0, __n_components - 1);
    s = s.rows(0, __n_components - 1);

    __signals = _X * V;
    __explained_variance = arma::square(s) / (X.n_rows - 1);
}

void clusterxx::PCA::fit(const arma::mat &X) {
    assert(!X.empty());
    __fit(X);
}

arma::mat clusterxx::PCA::fit_transform(const arma::mat &X) {
    assert(!X.empty());
    __fit(X);
    return __signals;
}

arma::vec clusterxx::PCA::get_explained_variance() const {
    assert(!__explained_variance.empty());
    return __explained_variance;
}

#endif
