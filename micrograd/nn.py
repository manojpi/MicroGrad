import random


class Neuron:
    def __init__(self, nin):
        self.w = [random.uniform(-1, 1) for _ in range(nin)] # weights
        self.b = random.uniform(-1, 1) # bias

    def __call__(self, x):
        act = sum((xi * wi for xi, wi in zip(x, self.w)), self.b)
        out = act.tanh()

        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        out = [neuron(x) for neuron in self.neurons]
        
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, nin, nouts: list):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x):

        for layer in self.layers:
            x = layer(x)
        
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]