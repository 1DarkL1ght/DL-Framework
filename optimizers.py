class SGD:
    def __init__(self, parameters, alpha=0.001):
        self.parameters = parameters
        self.alpha = alpha
        assert len(self.parameters) != 0, 'Optimizer got an empty parameters list'

    def zero(self):
        for param in self.parameters:
            param.grad.data *= 0

    def step(self):
        for param in self.parameters:
            param.data -= param.grad.data * self.alpha
