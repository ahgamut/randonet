# -*- coding: utf-8 -*-
"""
    randonet.unit
    ~~~~~~~~~~~~~

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
from collections import OrderedDict, namedtuple


class Unit(object):
    def __init__(self, fn, in_s=list(), out_s=list()):
        self.fn = fn
        self.in_shape = in_s
        self.out_shape = out_s

    def __str__(self):
        return str(self.fn)

    def __repr__(self):
        return str(self.fn)


class Factory(object):
    def __init__(self):
        self.template_fn = None
        self.params = None

    def __getattr__(self, key):
        try:
            return getattr(self.params, key)
        except Exception as e:
            return self.__dict__[key]

    def _render(self):
        param_dict = OrderedDict()
        for p in self.params:
            p(param_dict)
        fn = self.template_fn(**param_dict)
        return fn

    def _fix_inshape(self, _in_shape):
        return _in_shape

    def _get_outshape(self, fn, in_shape):
        return in_shape

    def _lock_kernel(self, fn, in_shape, out_shape):
        return fn

    def __call__(self, _in_shape, _out_shape=None):

        in_shape = self._fix_inshape(_in_shape)
        fn = self._render()
        if _out_shape is not None:
            fn = self._lock_kernel(fn, in_shape, _out_shape)
            out_shape = _out_shape
        else:
            out_shape = self._get_outshape(fn, in_shape)
        return Unit(fn, in_shape, out_shape)
