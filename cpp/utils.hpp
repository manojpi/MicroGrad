// C++17 port of micrograd/utils.py: emits Graphviz DOT text instead of using the graphviz package.
#pragma once
#include <iomanip>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include "engine.hpp"

inline std::pair<std::set<Node*>, std::set<std::pair<Node*, Node*>>> trace(const Value& root) {
    std::set<Node*> nodes;
    std::set<std::pair<Node*, Node*>> edges;
    std::function<void(Node*)> build = [&](Node* v) {
        if (nodes.insert(v).second)
            for (auto& c : v->prev) {
                edges.insert({c.get(), v});
                build(c.get());
            }
    };
    build(root.n.get());
    return {nodes, edges};
}

// Render with: dot -Tsvg graph.dot -o graph.svg
inline std::string draw_dot(const Value& root) {
    auto [nodes, edges] = trace(root);
    std::ostringstream os;
    os << std::fixed << std::setprecision(4) << "digraph {\n  rankdir=LR;\n";
    for (Node* n : nodes) {
        os << "  \"" << n << "\" [shape=record, label=\"{ " << n->label << " | data " << n->data << " | grad "
           << n->grad << " }\"];\n";
        if (!n->op.empty()) {
            os << "  \"" << n << n->op << "\" [label=\"" << n->op << "\"];\n";
            os << "  \"" << n << n->op << "\" -> \"" << n << "\";\n";
        }
    }
    for (auto& [a, b] : edges) os << "  \"" << a << "\" -> \"" << b << b->op << "\";\n";
    os << "}\n";
    return os.str();
}
