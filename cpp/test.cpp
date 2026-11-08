// Minimal gradient checks with plain asserts.
#undef NDEBUG
#include <cassert>
#include <cmath>
#include <iostream>
#include "nn.hpp"
#include "utils.hpp"

static bool near(double a, double b, double tol = 1e-6) { return std::fabs(a - b) < tol; }

int main() {
    // Hand-computed: L = (a*b + c) * f, a=2, b=-3, c=10, f=-2
    {
        Value a(2.0), b(-3.0), c(10.0), f(-2.0);
        Value L = (a * b + c) * f;
        L.backward();
        assert(near(L.data(), -8.0));
        assert(near(a.grad(), 6.0) && near(b.grad(), -4.0));
        assert(near(c.grad(), -2.0) && near(f.grad(), 4.0));
    }
    // Gradient accumulation: same node used twice.
    {
        Value a(3.0);
        Value b = a + a;
        b.backward();
        assert(near(a.grad(), 2.0));
    }
    // Same node on both sides of a product: d(a*a)/da = 2a.
    {
        Value a(3.0);
        (a * a).backward();
        assert(near(a.grad(), 6.0));
    }
    // Large-input tanh must not produce NaN.
    {
        Value a(400.0);
        Value t = a.tanh();
        t.backward();
        assert(near(t.data(), 1.0) && near(a.grad(), 0.0));
        assert(Value(1.0).pow(1e-7).n->op == "**1e-07");
    }
    // Every op vs. central-difference numerical gradient.
    {
        auto expr = [](const Value& x, const Value& y) {
            return ((x * y + 2.0) / (y - 4.0)).tanh() + (-x).exp() * y.pow(3) - 1.5 * x;
        };
        double x0 = 0.7, y0 = -1.3, h = 1e-6;
        Value x(x0), y(y0);
        expr(x, y).backward();
        double nx = (expr(x0 + h, y0).data() - expr(x0 - h, y0).data()) / (2 * h);
        double ny = (expr(x0, y0 + h).data() - expr(x0, y0 - h).data()) / (2 * h);
        assert(near(x.grad(), nx, 1e-5));
        assert(near(y.grad(), ny, 1e-5));
    }
    // nn: parameter counts and a numerical check on one weight.
    {
        rng().seed(1);
        MLP mlp(3, {4, 4, 1});
        auto params = mlp.parameters();
        assert(params.size() == 4 * 4 + 4 * 5 + 1 * 5);
        std::vector<Value> x = {2.0, 3.0, -1.0};
        mlp(x)[0].backward();
        Value w = mlp.layers[0].neurons[0].w[0];
        double g = w.grad(), w0 = w.data(), h = 1e-6;
        w.data() = w0 + h; double up = mlp(x)[0].data();
        w.data() = w0 - h; double dn = mlp(x)[0].data();
        w.data() = w0;
        assert(near(g, (up - dn) / (2 * h), 1e-5));
    }
    // utils: trace finds nodes/edges of a + b.
    {
        Value a(1.0), b(2.0);
        auto [nodes, edges] = trace(a + b);
        assert(nodes.size() == 3 && edges.size() == 2);
        assert(draw_dot(a + b).find("digraph") == 0);
    }
    std::cout << "All tests passed\n";
}
