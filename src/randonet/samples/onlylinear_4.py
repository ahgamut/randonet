# -*- coding: utf-8 -*-
"""
    onlylinear_4.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class OnlyLinear_4(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Linear(in_features=784, out_features=35, bias=True)
        self.f1 = nn.Linear(in_features=35, out_features=12, bias=True)
        self.f2 = nn.Linear(in_features=12, out_features=11, bias=True)
        self.f3 = nn.Linear(in_features=11, out_features=10, bias=True)

    def forward(self, *inputs):
        x = inputs[0]
        x = torch.view(x[0], 784)
        x = self.f0(x)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return x