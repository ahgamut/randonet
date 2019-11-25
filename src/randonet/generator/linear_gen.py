# -*- coding: utf-8 -*-
"""
    randonet.linear_gen
    ~~~~~~~~~~~~~~~~~~~

    Generate a sequential neural network with just linear layers

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
import os
from randonet.pytorch import Linear
from randonet.generator.param import IntParam


class OnlyLinear(object):
    def __init__(
        self, start_shape, stop_shape, min_features=10, max_features=256, depth=2
    ):
        self.start_shape = start_shape
        self.stop_shape = stop_shape
        self.max_features = 256
        self.depth = depth
        self.gtor = Linear()
        self.gtor.out_features.randomize(limits=(min_features, max_features))

    def generate(self):
        limits = self.gtor.out_features.limits
        unit_list = []
        for j in range(self.depth):
            if j == 0:
                in_shape = self.start_shape
            else:
                in_shape = unit_list[-1].out_shape
            if j == self.depth - 1:
                out_shape = self.stop_shape
            else:
                out_shape = None
            unit_list.append(self.gtor(in_shape, out_shape))
            self.gtor.out_features.limits = (limits[0], unit_list[-1].out_shape[0])

        self.gtor.out_features.limits = limits
        return unit_list

    def __call__(self, num_nets):
        ans = []
        for i in range(num_nets):
            name = "OnlyLinear_{}".format(i + 1)
            ans.append(dict(name=name, layers=self.generate()))
        return ans
