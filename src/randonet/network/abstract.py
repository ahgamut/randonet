# -*- coding: utf-8 -*-
"""
    randonet.abstract
    ~~~~~~~~~~~~~~~~~

    Abstract base class for network generation

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""


class AbstractNet(object):
    def __init__(self, start_shape, stop_shape, depth):
        self.start_shape = start_shape
        self.stop_shape = stop_shape
        self.depth = depth
        self.name = self.__class__.__name__
        self.layers = []

    def generate(self):
        raise NotImplementedError("Abstract Base Class")

    def __call__(self, num_nets, startnum=1):
        ans = []
        for i in range(num_nets):
            name = "{}_{}".format(self.name, i + startnum)
            ans.append(dict(name=name, layers=self.generate()))
        return ans
