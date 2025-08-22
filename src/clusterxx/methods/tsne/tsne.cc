#include "tsne.hpp"
#include "utils.hpp"
#include <random>

void clusterxx::TSNE::__fit(const std::vector<std::vector<double> > &X) {
    assert(__n_components < X[0].size());
    assert(__perplexity < X.size());

    // compute pairwise affinities
    std::vector<std::vector<double> > p_ji = compute_pairwise_affinities(X, __perplexity);
    std::vector<std::vector<double> > symmetrized = p_ji;
    for (size_t i = 0; i<p_ji.size(); i++) {
        for (size_t j = 0; j<p_ji[0].size(); j++) {
            symmetrized[i][j] = (p_ji[i][j] + p_ji[j][i]) / (2.0 * X.size());
        }
    }
    
    // sample initial solution from N(0, 1e-4)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1e-4);
    
    std::vector<std::vector<double> > solution(X.size(), std::vector<double>(__n_components));
    for (size_t i = 0; i<solution.size(); i++) {
        for (size_t j = 0; j<__n_components; j++) {
            solution[i][j] = dist(gen);
        }
    }

    std::vector<std::vector<std::vector<double> > > solution_hist;

    for (int i = 0; i<__max_iter; i++) {

        if (i == 0.25 * __max_iter) [[unlikely]] {
            // increase momentum after 1/4 of the iterations
            __momentum = 0.8;
        }

        // compute low dimensional affinities(q_ij)
        std::vector<std::vector<double> > q_ij = compute_low_dim_affinities(solution);
        
        // compute gradient
        std::vector<std::vector<double> > gradients =
            kullback_leibler_gradient(symmetrized, q_ij, solution);
        
        // update solution
        for (size_t j = 0; j<solution.size(); j++) {
            for (size_t k = 0; k<solution[j].size(); k++) {

                double momentum_factor = 0.0, previous_factor = 0.0;
                if (i > 0) [[likely]] {
                    previous_factor = solution_hist.back()[j][k];
                }
                if (i > 1) [[likely]] {
                    momentum_factor = solution_hist.back()[j][k] - solution_hist[solution_hist.size() - 2][j][k];
                }
                solution[j][k] = previous_factor + __learning_rate * gradients[j][k] + __momentum * momentum_factor;
            }
        }
        solution_hist.push_back(solution);
    }
    
    for (auto &v: solution) { __features.push_back(v); }
    __shape.first = solution.size();
    __shape.second = __n_components;
}

void clusterxx::TSNE::fit(const std::vector<std::vector<double> > X) {
    __fit(X);
}


std::vector<std::vector<double> > clusterxx::TSNE::fit_transform(const std::vector<std::vector<double> > X) {
    __fit(X);
    return __features;
}

std::pair<int, int> clusterxx::TSNE::get_shape() {
    return __shape;
}

std::vector<std::vector<double> > clusterxx::TSNE::get_features() {
    return __features;
}
