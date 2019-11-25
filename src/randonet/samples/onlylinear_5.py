# -*- coding: utf-8 -*-
"""
    onlylinear_5.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class OnlyLinear_5(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Linear(in_features=784, out_features=26, bias=True)
        self.f1 = nn.Linear(in_features=26, out_features=12, bias=True)
        self.f2 = nn.Linear(in_features=12, out_features=11, bias=True)
        self.f3 = nn.Linear(in_features=11, out_features=10, bias=True)
        self.f4 = nn.Linear(in_features=10, out_features=10, bias=True)
        self.f5 = nn.Linear(in_features=10, out_features=10, bias=True)
        self.f6 = nn.Linear(in_features=10, out_features=10, bias=True)
        self.f7 = nn.Linear(in_features=10, out_features=10, bias=True)

    def forward(self, *inputs):
        x = inputs[0]
        x = torch.view(x[0], 784)
        x = self.f0(x)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        return x