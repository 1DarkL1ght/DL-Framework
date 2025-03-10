import numpy as np
import random


class Tensor:
  def __init__(self, data, autograd=False, creators=None, creation_op=None):
    self.data = np.array(data)
    self.h_params = {}
    self.autograd = autograd
    self.grad = None
    self.creators = creators
    self.children = {}
    self.creation_op = creation_op
    self.id = random.randint(0, 100000)
    if self.creators is not None:
      for c in creators:
        if self.id not in c.children:
          c.children[self.id] = 1
        else:
          c.children[self.id] += 1

  def __add__(self, other):
    if self.autograd and other.autograd:
      return Tensor(self.data + other.data, autograd=True, creators=[self, other], creation_op='add')
    return Tensor(self.data + other.data)

  def __neg__(self):
    if self.autograd:
      return Tensor(self.data * -1, autograd=True, creators=[self], creation_op='neg')
    return Tensor(self.data * -1)

  def __sub__(self, other):
    if self.autograd and other.autograd:
      return Tensor(self.data - other.data, autograd=True, creators=[self, other], creation_op='sub')
    return Tensor(self.data - other.data)

  def __mul__(self, other):
    if self.autograd and other.autograd:
      return Tensor(self.data * other.data, autograd=True, creators=[self, other], creation_op='mul')
    return Tensor(self.data * other.data)

  def sum(self, dim):
    if self.autograd:
      return Tensor(self.data.sum(dim), autograd=True, creators=[self], creation_op='sum_'+str(dim))
    return Tensor(self.data.sum(dim))

  def expand(self, dim, copies):
    trans_cmd = list(range(0, len(self.data.shape)))
    trans_cmd.insert(dim, len(self.data.shape))
    new_shape = list(self.data.shape) + [copies]
    new_data = self.data.repeat(copies).reshape(new_shape)
    new_data = new_data.transpose(trans_cmd)

    if self.autograd:
      return Tensor(new_data, autograd=True, creators=[self], creation_op='expand_'+str(dim))
    return Tensor(new_data)

  def transpose(self):
    if self.autograd:
      return Tensor(self.data.transpose(), autograd=True, creators=[self], creation_op='transpose')
    return Tensor(self.data.transpose())

  def mm(self, other):
    if self.autograd:
      return Tensor(self.data.dot(other.data), autograd=True, creators=[self, other], creation_op='mm')
    return Tensor(self.data.dot(other.data))

  def relu_der(self, x):
    return np.where(x > 0, 1, 0)

  def leakyrelu_der(self, x):
    return np.where(x > 0, x, self.h_params['leaky'] * x)

  def reshape(self, shape):
    if self.autograd:
      return Tensor(self.data.reshape(shape), autograd=True, creators=self.creators, creation_op=self.creation_op)
    return Tensor(self.data.reshape(shape))

  def all_children_grads_counted(self):
    for id, cnt in self.children.items():
      if cnt != 0:
        return False
      return True

  def backward(self, grad, grad_origin=None):
    #print('grad=', grad.data)
    print(f"Backwarding. Autograd = {self.autograd}.", end=' ')
    print(f"Creators = {[creator.id for creator in self.creators]}, creation_op = {self.creation_op}, {self.id}")
    if self.autograd:
      if grad_origin is not None:
        if self.children[grad_origin.id] == 0:
          raise Exception("cannot backprop more than once")
        else:
          self.children[grad_origin.id] -= 1

      if self.grad is None:
        # print('grad adding')
        self.grad = grad
      else:
        self.grad += grad
      # print('self grad', self.grad)

      assert grad.autograd == False
      if self.creators is not None and (self.all_children_grads_counted() or grad_origin is None):
        if self.creation_op == 'add':
          self.creators[0].backward(grad)
          self.creators[1].backward(grad)
        elif self.creation_op == 'neg':
          self.creators[0].backward(self.grad.__neg__())
        elif self.creation_op == 'sub':
          grad1 = Tensor(self.grad.data)
          grad2 = Tensor(self.grad.__neg__().data)
          self.creators[0].backward(grad1)
          self.creators[1].backward(grad2)
        elif self.creation_op == 'mul':
          grad1 = self.grad * self.creators[1]
          grad2 = self.grad * self.creators[0]
          self.creators[0].backward(grad1)
          self.creators[1].backward(grad2)
        elif self.creation_op == 'transpose':
          self.creators[0].backward(self.grad.transpose())
        elif self.creation_op == 'mm':
          act = self.creators[0]
          weights = self.creators[1]
          new = self.grad.mm(weights.transpose())
          act.backward(new)
          new = self.grad.transpose().mm(act).transpose()
          weights.backward(new)
        elif 'sum' in self.creation_op:
          dim = int(self.creation_op.split('_')[1])
          ds = self.creators[0].data.shape[dim]
          self.creators[0].backward(self.grad.expand(dim, ds))
        elif 'expand' in self.creation_op:
          dim = int(self.creation_op.split('_')[1])
          self.creators[0].backward(self.grad.sum(dim))
        elif self.creation_op == 'relu':
          self.grad.data *= self.relu_der(self.data)
          self.creators[0].backward(self.grad)
        elif self.creation_op == 'sigmoid':
          ones = Tensor(np.ones_like(self.grad.data))
          self.grad *= (self * (ones - self))
          self.creators[0].backward(self.grad)
        elif self.creation_op == 'leaky':
          self.grad.data *= self.leakyrelu_der(self.data)
          self.creators[0].backward(self.grad)
        elif self.creation_op == 'tanh':
          ones = Tensor(np.ones_like(self.grad.data))
          self.creators[0].backward(self.grad * (ones - (self * self)))