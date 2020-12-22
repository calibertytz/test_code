from torch.nn.modules.batchnorm import BatchNorm1d
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import numpy as np
from torch.nn import init


class GhostNormalization(nn.Module):
    def __init__(self, input_dim, virtual_batch_size=64, momentum=0.01):
        super(GhostNormalization, self).__init__()

        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = BatchNorm1d(self.input_dim, momentum=momentum, affine=False)
        self.weight = Parameter(torch.Tensor(self.input_dim))
        self.bias = Parameter(torch.Tensor(self.input_dim))

        init.uniform_(self.bias)
        init.uniform_(self.weight)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [(self.weight * self.bn(x_) + self.bias) for x_ in chunks]
        return torch.cat(res, dim=0)

if __name__ == '__main__':
    gn = GhostNormalization(input_dim=10)
    x = torch.rand((10, 10), requires_grad=True)
    y = gn(x)
    assert x.size() == y.size()
    print('done!')