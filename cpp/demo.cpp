// Mirrors the small-dataset training loop in notebook/micrograd.ipynb.
#include <iostream>
#include "nn.hpp"

int main() {
    rng().seed(42);
    std::vector<std::vector<Value>> xs = {
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0},
    };
    std::vector<double> ys = {1.0, -1.0, -1.0, 1.0};  // desired targets
    MLP mlp(3, {4, 4, 1});

    for (int k = 0; k < 20; ++k) {
        // forward pass
        Value loss(0.0);
        for (size_t i = 0; i < xs.size(); ++i) {
            Value pred = mlp(xs[i])[0];
            loss += (pred - ys[i]).pow(2);
        }
        // backward pass
        for (auto& p : mlp.parameters()) p.grad() = 0.0;
        loss.backward();
        // update
        for (auto& p : mlp.parameters()) p.data() += -0.01 * p.grad();
        std::cout << k << " " << loss.data() << "\n";
    }
    std::cout << "ypred:";
    for (auto& x : xs) std::cout << " " << mlp(x)[0].data();
    std::cout << "\n";
}
