# -*- coding: utf-8 -*-
"""
    randonet.res_gen
    ~~~~~~~~~~~~~~~~

    Use Resnet BasicBlocks as layers

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
from randonet.network.abstract import AbstractNet as _Net
from randonet.network.c2l_gen import Conv2dThenLinear


class ResNetStyle(_Net):
    pass
