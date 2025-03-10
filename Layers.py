from Tensor import Tensor
import numpy as np


class Layer:
  def __init__(self):
    self.parameters = []

  def get_parameters(self):
    return self.parameters


class Activation(Layer):
  def __init__(self):
    super().__init__()


class Sequential(Layer):
  def __init__(self, layers):
    super().__init__()
    self.layers = layers
    self.parameters = []
    for i in range(len(self.layers)):
      if isinstance(self.layers[i], Layer):
        for param in self.layers[i].get_parameters():
            print('param =', param)
            if param is not None:
                self.parameters.append(param)

  def forward(self, x):
    for i in range(len(self.layers)):
      if isinstance(self.layers[i], Layer):
        x = self.layers[i].forward(x)
    return x


class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.reshape((-1, np.prod(x.data.shape[1:])))


class Linear(Layer):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.weights = Tensor(np.random.randn(in_features, out_features) * np.sqrt(2 / in_features), autograd=True)
    self.bias = Tensor(np.zeros((out_features)), autograd=True)
    self.parameters.append(self.weights)
    self.parameters.append(self.bias)

  def forward(self, x):
    return x.mm(self.weights) + self.bias.expand(0, len(x.data))


class Identity(Layer):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return x


class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if x.autograd:
            return Tensor(x.data * (x.data > 0), autograd=True, creators=[x], creation_op='relu')
        return Tensor(x.data * (x.data > 0))


class LeakyReLU(Activation):
  def __init__(self, a):
    super().__init__()
    self.a = a

  def forward(self, x):
    if x.autograd:
       t = Tensor(np.where(x.data > 0, x.data, x.data * self.a), autograd=True, creators=[x], creation_op='leaky')
       t.h_params['leaky'] = self.a
       return t
    t = Tensor(np.where(x.data > 0, x.data, x.data * self.a))
    t.h_params['leaky'] = self.a
    return t


class Sigmoid(Activation):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    if x.autograd:
      return Tensor(1 / (1 + np.exp(-x.data)),
                            autograd=True,
                            creators=[x],
                            creation_op="sigmoid")
    return Tensor(1 / (1 + np.exp(-x.data)))


class Tanh(Activation):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    if x.autograd:
      return Tensor(np.tanh(x.data), autograd=True, creators=[x], creation_op='tanh')
    return Tensor(np.tanh(x.data))


class Softmax(Activation):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    softmax = np.empty_like(x.data)
    for i in range(x.data.shape[0]):
      sum = np.sum(np.exp(x.data[i]))
      for j in range(x.data.shape[1]):
        softmax[i][j] = np.exp(x.data[i][j]) / sum
    if x.autograd:
      return Tensor(softmax, autograd=True, creators=[x], creation_op='softmax')
    return Tensor(softmax)


class Conv2d(Layer):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
    super().__init__()
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.conved_h: int
    self.conved_w: int
    try:
      self.kernel_x, self.kernel_y = self.kernel_size[1], self.kernel_size[0]
    except:
      self.kernel_x = self.kernel_y = self.kernel_size
    self.lin = Linear(self.kernel_x * self.kernel_y * self.in_channels, self.out_channels)
    self.parameters = self.lin.get_parameters()
    #print('kernel_size=', self.kernel_size, 'stride=', self.stride, 'padding=', self.padding, 'x step=', self.kernel_x, 'y step=', self.kernel_y)

  def forward(self, x):
    self.conved_h = (x.data.shape[1] - self.kernel_y) // self.stride + 1
    self.conved_w = (x.data.shape[2] - self.kernel_x) // self.stride + 1
    conved = np.empty((x.data.shape[0], self.conved_h, self.conved_w, self.out_channels))
    for i in range(0, self.conved_h):
      for j in range(0, self.conved_w):
        conv = x.data[:, i * self.stride:i * self.stride + self.kernel_x, j * self.stride:j * self.stride + self.kernel_y]
        input = Tensor(conv.reshape(-1, conv.shape[1] * conv.shape[2] * conv.shape[3]))
        output = self.lin.forward(input)
        for k in range(output.data.shape[0]):
          conved[k][i][j] = output.data[k]
    if x.autograd:
      return Tensor(conved, autograd=True, creators=output.creators, creation_op=output.creation_op)
    return Tensor(conved)


class MaxPool2d(Layer):
    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conved_h: int
        self.conved_w: int
        try:
            self.kernel_x, self.kernel_y = self.kernel_size[1], self.kernel_size[0]
        except:
            self.kernel_x = self.kernel_y = self.kernel_size
        # print('kernel_size=', self.kernel_size, 'stride=', self.stride, 'padding=', self.padding, 'x step=', self.kernel_x, 'y step=', self.kernel_y)

    def forward(self, x):
        self.conved_h = (x.data.shape[1] - self.kernel_y) // self.stride + 1
        self.conved_w = (x.data.shape[2] - self.kernel_x) // self.stride + 1
        conved = np.empty((x.data.shape[0], self.conved_h, self.conved_w, x.data.shape[3]))
        for i in range(0, self.conved_h):
            for j in range(0, self.conved_w):
                conv = x.data[:, i * self.stride:i * self.stride + self.kernel_x,
                       j * self.stride:j * self.stride + self.kernel_y]
                conv = np.max(conv, axis=(1, 2))
                for k in range(conved.shape[0]):
                    conved[k][i][j] = conv[k]
        if x.autograd:
            return Tensor(conved, autograd=True, creators=x.creators, creation_op=x.creation_op)
        return Tensor(conved)

