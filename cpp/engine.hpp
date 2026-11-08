// C++17 port of micrograd/engine.py
#pragma once
#include <cmath>
#include <functional>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

struct Node {
    double data;
    double grad = 0.0;
    std::function<void()> backward = [] {};
    std::vector<std::shared_ptr<Node>> prev;  // children keep the graph alive
    std::string op;
    std::string label;
};

// Value is a cheap handle to a shared graph node, so copies alias the same node
// (matches Python reference semantics).
class Value {
public:
    std::shared_ptr<Node> n;

    Value(double data = 0.0, std::vector<Value> children = {}, std::string op = "", std::string label = "")
        : n(std::make_shared<Node>()) {
        n->data = data;
        for (auto& c : children) n->prev.push_back(c.n);
        n->op = std::move(op);
        n->label = std::move(label);
    }

    double& data() const { return n->data; }
    double& grad() const { return n->grad; }
    std::string& label() const { return n->label; }

    // Closures capture raw pointers: `out` owns its closure and (via prev) its children,
    // so the pointers stay valid as long as `out` lives, and no shared_ptr cycle is formed.
    friend Value operator+(const Value& a, const Value& b) {
        Value out(a.data() + b.data(), {a, b}, "+");
        Node *x = a.n.get(), *y = b.n.get(), *o = out.n.get();
        out.n->backward = [x, y, o] {
            x->grad += 1.0 * o->grad;
            y->grad += 1.0 * o->grad;
        };
        return out;
    }

    friend Value operator*(const Value& a, const Value& b) {
        Value out(a.data() * b.data(), {a, b}, "*");
        Node *x = a.n.get(), *y = b.n.get(), *o = out.n.get();
        out.n->backward = [x, y, o] {
            x->grad += y->data * o->grad;
            y->grad += x->data * o->grad;
        };
        return out;
    }

    // Python's ** with a numeric exponent only.
    Value pow(double k) const {
        std::ostringstream ks;
        ks << "**" << k;
        Value out(std::pow(data(), k), {*this}, ks.str());
        Node *x = n.get(), *o = out.n.get();
        out.n->backward = [x, o, k] { x->grad += k * std::pow(x->data, k - 1) * o->grad; };
        return out;
    }

    Value tanh() const {
        double t = std::tanh(data());
        Value out(t, {*this}, "tanh");
        Node *s = n.get(), *o = out.n.get();
        out.n->backward = [s, o, t] { s->grad += (1 - t * t) * o->grad; };
        return out;
    }

    Value exp() const {
        Value out(std::exp(data()), {*this}, "exp");
        Node *s = n.get(), *o = out.n.get();
        out.n->backward = [s, o] { s->grad += o->grad * o->data; };
        return out;
    }

    friend Value operator-(const Value& a) { return a * -1.0; }
    friend Value operator-(const Value& a, const Value& b) { return a + (-b); }
    friend Value operator/(const Value& a, const Value& b) { return a * b.pow(-1); }
    Value& operator+=(const Value& b) { return *this = *this + b; }

    void backward() const {
        // topological sort for managing gradient dependencies
        std::vector<Node*> topo;
        std::unordered_set<Node*> visited;
        std::function<void(Node*)> build = [&](Node* v) {
            if (visited.insert(v).second) {
                for (auto& c : v->prev) build(c.get());
                topo.push_back(v);
            }
        };
        build(n.get());
        n->grad = 1.0;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) (*it)->backward();
    }

    friend std::ostream& operator<<(std::ostream& os, const Value& v) {
        return os << "Value(data=" << v.data() << ")";
    }
};

// Mixed scalar overloads (Python's __radd__/__rmul__ and implicit Value(other) wrapping).
inline Value operator+(const Value& a, double b) { return a + Value(b); }
inline Value operator+(double a, const Value& b) { return Value(a) + b; }
inline Value operator*(const Value& a, double b) { return a * Value(b); }
inline Value operator*(double a, const Value& b) { return Value(a) * b; }
inline Value operator-(const Value& a, double b) { return a - Value(b); }
inline Value operator-(double a, const Value& b) { return Value(a) - b; }
inline Value operator/(const Value& a, double b) { return a / Value(b); }
inline Value operator/(double a, const Value& b) { return Value(a) / b; }
