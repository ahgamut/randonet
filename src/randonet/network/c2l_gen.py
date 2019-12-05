# -*- coding: utf-8 -*-
"""
    randonet.c2l_gen
    ~~~~~~~~~~~~~~~~

    Convolution followed by linear layers

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
from randonet.network.abstract import AbstractNet as _Net
from randonet.generator.param import IntParam
from randonet.network.conv_gen import Conv2dAC, Conv1dAC, Conv3dAC
from randonet.network.linear_gen import LinearAC


class ConvThenLinear(_Net):
    def __init__(self, start_shape, stop_shape, depth=2, conv_part=Conv1dAC):
        _Net.__init__(self, start_shape, stop_shape, depth)
        self.conv_part = conv_part(
            start_shape=start_shape, stop_shape=stop_shape, depth=depth, bias_prob=0.3
        )
        self.lin_part = LinearAC(
            start_shape=None, stop_shape=(10,), depth=depth, bias_prob=0.3
        )
        self.change_num = IntParam(name="")

    def change_when(self, x):
        self.change_num.unrandomize(val=x)

    def change_random(self):
        self.change_num.randomize(limits=(1, self.depth - 1))

    def generate(self):
        self.change_random()
        t = self.change_num.value
        self.lin_part.depth = self.depth - t
        cp = self.conv_part.generate()[:t]
        self.lin_part.start_shape = cp[-1].out_shape
        lp = self.lin_part.generate()
        return cp + lp


class Conv1dThenLinear(ConvThenLinear):
    def __init__(self, start_shape=(16, 49), stop_shape=(10, 1), depth=0):
        ConvThenLinear.__init__(
            self, start_shape, stop_shape, depth, conv_part=Conv1dAC
        )


class Conv2dThenLinear(ConvThenLinear):
    def __init__(self, start_shape=(1, 28, 28), stop_shape=(10, 1, 1), depth=0):
        ConvThenLinear.__init__(
            self, start_shape, stop_shape, depth, conv_part=Conv2dAC
        )


class Conv3dThenLinear(ConvThenLinear):
    def __init__(self, start_shape=(1, 16, 7, 7), stop_shape=(10, 1, 1, 1), depth=0):
        ConvThenLinear.__init__(
            self, start_shape, stop_shape, depth, conv_part=Conv3dAC
        )
