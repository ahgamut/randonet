# -*- coding: utf-8 -*-
"""
    randonet.res_gen
    ~~~~~~~~~~~~~~~~

    Use Resnet BasicBlocks as layers

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
from collections import namedtuple
from randonet.generator.unit import Factory as _Factory
from randonet.network.abstract import AbstractNet as _Net
from randonet.network.c2l_gen import Conv2dThenLinear
from randonet.network.conv_gen import Conv2dAC
from randonet.network.linear_gen import LinearAC
from randonet.generator.param import IntParam


class BasicBlock(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("BasicBlock", ["inplanes", "planes"])
        self.params = self.template_fn(
            inplanes=IntParam(name="inplanes", default=1),
            planes=IntParam(name="planes", default=1),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v

    def _fix_inshape(self, _in_shape):
        self.inplanes.val = _in_shape[0]
        self.planes.val = _in_shape[0]
        return _in_shape

    def _lock_kernel(self, fn, in_shape, out_shape):
        return fn._replace(inplanes=in_shape[0], planes=in_shape[0])


class ResNetStyle(_Net):
    def __init__(self, start_shape, stop_shape, depth=3):
        _Net.__init__(self, start_shape, stop_shape, depth)
        self.layers.append(Conv2dAC(start_shape, [10, 1, 1], depth))
        self.layers.append(Conv2dThenLinear(start_shape, [10, 1, 1], depth))
        self.change_num = IntParam("", default=1)

    def generate(self):
        z1 = Conv2dAC.ac.val
        Conv2dAC.ac.val = Conv2dAC.ac.choices[2]
        z2 = LinearAC.ac.val
        LinearAC.ac.val = LinearAC.ac.choices[2]
        self.change_num.randomize(limits=(1, self.depth - 2 * self.depth // 3))
        t = self.change_num.value
        cp = self.layers[0].generate()[:t]

        skip_gen = BasicBlock()
        self.change_num.randomize(
            limits=(1, max([1, self.depth - (t + self.depth // 3)]))
        )
        t2 = self.change_num.value

        sp = []
        for i in range(t2):
            sp.append(skip_gen(_in_shape=cp[-1].out_shape))

        self.change_num.randomize(limits=(2, max([2, self.depth - (t + t2)])))
        t3 = self.change_num.value
        self.layers[1].start_shape = sp[-1].out_shape
        self.layers[1].conv_part.start_shape = sp[-1].out_shape
        self.layers[1].depth = t3
        lp = self.layers[1].generate()

        Conv2dAC.ac.val = z1
        LinearAC.ac.val = z2
        return cp + sp + lp
