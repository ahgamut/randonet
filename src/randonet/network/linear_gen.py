# -*- coding: utf-8 -*-
"""
    randonet.linear_gen
    ~~~~~~~~~~~~~~~~~~~

    Generate a sequential neural network with linear layers

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
from randonet.pytorch import Linear
from randonet.network import ActivationParam
from randonet.generator.param import BinaryParam
from randonet.network.abstract import AbstractNet as _Net


class LinearOnly(_Net):
    def __init__(
        self,
        start_shape,
        stop_shape,
        min_features=10,
        max_features=128,
        depth=2,
        bias_prob=0.0,
    ):
        _Net.__init__(self, start_shape, stop_shape, depth)
        self.layers = [Linear(bias=False)]
        self.limits = (min_features, max_features)
        if bias_prob > 0:
            self.layers[0].bias.randomize(true_prob=bias_prob)
        self.layers[0].out_features.randomize(limits=self.limits)

    def generate(self):
        limits = self.limits
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
            unit_list.append(self.layers[0](in_shape, out_shape))
            self.layers[0].out_features.limits = (limits[0], unit_list[-1].out_shape[0])

        self.layers[0].out_features.limits = limits
        return unit_list

    def __call__(self, num_nets, startnum=1):
        if self.layers[0].bias.is_random:
            if "Only" in self.name:
                self.name = self.name.replace("Only", "Bias")
            else:
                self.name = self.name + "Bias"
        _Net.__call__(self, num_nets, startnum)


class LinearAC(LinearOnly):
    def __init__(
        self,
        start_shape,
        stop_shape,
        min_features=10,
        max_features=128,
        depth=2,
        bias_prob=0.0,
    ):
        LinearOnly.__init__(
            self, start_shape, stop_shape, min_features, max_features, depth, bias_prob
        )
        self.ac = ActivationParam()
        self.use_ac = BinaryParam()
        self.use_ac.randomize(true_prob=0.7)

    def generate(self):
        limits = self.limits
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
            unit_list.append(self.layers[0](in_shape, out_shape))
            if self.use_ac.value:
                unit_list.append(self.ac.val(out_shape, out_shape))
            self.layers[0].out_features.limits = (limits[0], unit_list[-1].out_shape[0])

        self.layers[0].out_features.limits = limits
        return unit_list

    def __call__(self, num_nets, startnum=1):
        self.name = "Linear{}".format(self.ac.val.__class__.__name__)
        return LinearOnly.__call__(self, num_nets, startnum)
