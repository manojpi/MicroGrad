# micrograd (C++17)

Header-only port of `micrograd/` (`engine.hpp`, `nn.hpp`, `utils.hpp`). Standard library only.

## Build with CMake

```sh
cmake -S cpp -B cpp/build && cmake --build cpp/build
./cpp/build/demo
./cpp/build/micrograd_test   # or: ctest --test-dir cpp/build
```

## Build with a plain compiler

```sh
mkdir -p cpp/build
c++ -std=c++17 -Wall -Wextra -O2 cpp/demo.cpp -o cpp/build/demo
c++ -std=c++17 -Wall -Wextra -O2 cpp/test.cpp -o cpp/build/micrograd_test
```

## Differences from Python

- `x ** k` is `x.pow(k)`; `Layer`/`MLP` always return `std::vector<Value>` (use `[0]` for one output).
- `Value` exposes `data()`, `grad()` and `label()` as mutable references.
- Very deep graphs (around 100k nodes) overflow the stack via the recursive topo sort and `shared_ptr` destruction; fine at micrograd scale.
- `draw_dot` returns Graphviz DOT text; render with `dot -Tsvg`.
- `Neuron` weights are `Value`s (as in the notebook); `nn.py` currently stores plain floats.
