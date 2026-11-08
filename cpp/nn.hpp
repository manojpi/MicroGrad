// C++17 port of micrograd/nn.py
#pragma once
#include <random>
#include <vector>
#include "engine.hpp"

// Shared RNG; seed it for reproducible runs.
inline std::mt19937& rng() {
    static std::mt19937 g(std::random_device{}());
    return g;
}
inline double uniform(double lo, double hi) { return std::uniform_real_distribution<double>(lo, hi)(rng()); }

class Neuron {
public:
    std::vector<Value> w;  // weights
    Value b;               // bias

    explicit Neuron(int nin) : b(uniform(-1, 1)) {
        for (int i = 0; i < nin; ++i) w.emplace_back(uniform(-1, 1));
    }

    Value operator()(const std::vector<Value>& x) const {
        Value act = b;
        for (size_t i = 0; i < w.size() && i < x.size(); ++i) act += x[i] * w[i];  // zip semantics
        return act.tanh();
    }

    std::vector<Value> parameters() const {
        auto p = w;
        p.push_back(b);
        return p;
    }
};

class Layer {
public:
    std::vector<Neuron> neurons;

    Layer(int nin, int nout) {
        for (int i = 0; i < nout; ++i) neurons.emplace_back(nin);
    }

    // shortcut: always returns a vector (Python unwraps size-1 outputs); use [0] for a scalar.
    std::vector<Value> operator()(const std::vector<Value>& x) const {
        std::vector<Value> out;
        for (auto& n : neurons) out.push_back(n(x));
        return out;
    }

    std::vector<Value> parameters() const {
        std::vector<Value> p;
        for (auto& n : neurons)
            for (auto& v : n.parameters()) p.push_back(v);
        return p;
    }
};

class MLP {
public:
    std::vector<Layer> layers;

    MLP(int nin, const std::vector<int>& nouts) {
        std::vector<int> sz{nin};
        sz.insert(sz.end(), nouts.begin(), nouts.end());
        for (size_t i = 0; i < nouts.size(); ++i) layers.emplace_back(sz[i], sz[i + 1]);
    }

    std::vector<Value> operator()(std::vector<Value> x) const {
        for (auto& l : layers) x = l(x);
        return x;
    }

    std::vector<Value> parameters() const {
        std::vector<Value> p;
        for (auto& l : layers)
            for (auto& v : l.parameters()) p.push_back(v);
        return p;
    }
};
