import numpy as np
import pandas as pd
from Tensor import Tensor
from Layers import *
from optimizers import SGD


def one_hot(x):
    empty = np.zeros((20000, 10))
    for i in range(len(x)):
        empty[i][x[i]] = 1
    y = Tensor(empty, autograd=True)
    return y


df = pd.read_csv('mnist_train_small.csv', header=None)
y = df[df.columns[0]].values.reshape((-1, 1))
y = one_hot(y)
data = Tensor(df[df.columns[1:]].values.reshape((-1, int(np.sqrt(df[df.columns[1:]].values.shape[1])), int(np.sqrt(df[df.columns[1:]].values.shape[1])), 1)) / 255, autograd=True)
model = Sequential([
                        #Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=0),
                        #MaxPool2d(2, 2),
                        #ReLU(),
                        #Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=0),
                        #MaxPool2d(2, 2),
                        #ReLU(),
                        #Conv2d(in_channels=8, out_channels=8, kernel_size=2, stride=1, padding=0),
                        #MaxPool2d(2, 2),
                        #ReLU(),
                        Flatten(),
                        Linear(784, 10),
                        Softmax()
    ])
optim = SGD(parameters=model.get_parameters(), alpha=0.01)

if __name__ == "__main__":
    print(f'x shape is {data.data.shape}, y shape is {y.data.shape}')
    for i in range(10):
      print(f'i={i}')
      pred = model.forward(data)
      print('pred + y', sum(pred.data[0]), sum(y.data[0]))
      loss = ((pred - y) * (pred - y)).sum(0)
      print('loss=', loss.data[0])
      loss.backward(Tensor(np.ones_like(loss.data)))
      optim.step()
      optim.zero()
