# -*- coding: utf-8 -*-
"""
    randonet.conv_gen
    ~~~~~~~~~~~~~~~~~

    Generate a sequential neural network only Convolution layers

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
from randonet.pytorch import Conv1d, Conv2d, Conv3d
from randonet.network.activation import ActivationParam
from randonet.network.abstract import AbstractNet as _Net
from randonet.generator.param import BinaryParam


class ConvOnly(_Net):
    def __init__(self, start_shape, stop_shape, depth, convclass, bias_prob=0):
        _Net.__init__(self, start_shape, stop_shape, depth)
        self.layers = [convclass()]
        self.layers[0].out_channels.randomize(limits=(self.stop_shape[0], 64))
        if bias_prob > 0:
            self.layers[0].bias.randomize(true_prob=bias_prob)

    def set_limits(self, **limits):
        for k, v in limits.items():
            t = getattr(self.layers[0].params, k)
            t.randomize(**v)

    def generate(self):
        t = len(self.start_shape) - 1
        self.layers[0].kernel_size.randomize(
            limits=((1,) * t, (min(self.start_shape[1:]),) * t)
        )
        limits = self.layers[0].kernel_size.limits
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
            out_shape = unit_list[-1].out_shape
            t = len(out_shape) - 1
            self.layers[0].kernel_size.randomize(
                limits=((1,) * t, (min(out_shape[1:]),) * t)
            )
        self.layers[0].kernel_size.randomize(limits=limits)
        return unit_list

    def __call__(self, num_nets, startnum=1):
        t = self.layers[0].bias
        if t.is_random and t.true_prob > 0:
            if "Only" in self.name:
                self.name = self.name.replace("Only", "Bias")
            elif "Bias" not in self.name:
                self.name = self.name + "Bias"
        return _Net.__call__(self, num_nets, startnum)


class ConvAC(ConvOnly):
    ac = ActivationParam()

    def __init__(self, start_shape, stop_shape, depth, convclass, bias_prob):
        ConvOnly.__init__(self, start_shape, stop_shape, depth, convclass, bias_prob)
        self.use_ac = BinaryParam(name="")
        self.use_ac.randomize(true_prob=0.7)

    def generate(self):
        t = len(self.start_shape) - 1
        self.layers[0].kernel_size.randomize(
            limits=((1,) * t, (min(self.start_shape[1:]),) * t)
        )
        limits = self.layers[0].kernel_size.limits
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
            if self.use_ac.value and out_shape is None:
                unit_list.append(
                    self.ac.val(unit_list[-1].out_shape, unit_list[-1].out_shape)
                )
            out_shape = unit_list[-1].out_shape
            t = len(out_shape) - 1
            self.layers[0].kernel_size.randomize(
                limits=((1,) * t, (min(out_shape[1:]),) * t)
            )
        self.layers[0].kernel_size.randomize(limits=limits)
        return unit_list

    def __call__(self, num_nets, startnum=1):
        self.name = "{}".format(
            self.__class__.__name__.replace("AC", self.ac.val.__class__.__name__)
        )
        return _Net.__call__(self, num_nets, startnum)


class Conv1dOnly(ConvOnly):
    def __init__(self, start_shape=(16, 49), stop_shape=(10, 1), depth=2, bias_prob=0):
        ConvOnly.__init__(
            self, start_shape, stop_shape, depth, convclass=Conv1d, bias_prob=bias_prob
        )


class Conv2dOnly(ConvOnly):
    def __init__(
        self, start_shape=(1, 28, 28), stop_shape=(10, 1, 1), depth=2, bias_prob=0
    ):
        ConvOnly.__init__(
            self, start_shape, stop_shape, depth, convclass=Conv2d, bias_prob=bias_prob
        )


class Conv3dOnly(ConvOnly):
    def __init__(
        self, start_shape=(1, 16, 7, 7), stop_shape=(10, 1, 1, 1), depth=2, bias_prob=0
    ):
        ConvOnly.__init__(
            self, start_shape, stop_shape, depth, convclass=Conv3d, bias_prob=bias_prob
        )


class Conv1dAC(ConvAC):
    def __init__(self, start_shape=(16, 49), stop_shape=(10, 1), depth=2, bias_prob=0):
        ConvAC.__init__(
            self, start_shape, stop_shape, depth, convclass=Conv1d, bias_prob=bias_prob
        )


class Conv2dAC(ConvAC):
    def __init__(
        self, start_shape=(1, 28, 28), stop_shape=(10, 1, 1), depth=2, bias_prob=0
    ):
        ConvAC.__init__(
            self, start_shape, stop_shape, depth, convclass=Conv2d, bias_prob=bias_prob
        )


class Conv3dAC(ConvAC):
    def __init__(
        self, start_shape=(1, 16, 7, 7), stop_shape=(10, 1, 1, 1), depth=2, bias_prob=0
    ):
        ConvAC.__init__(
            self, start_shape, stop_shape, depth, convclass=Conv3d, bias_prob=bias_prob
        )
