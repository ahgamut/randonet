from randonet.generator.param import (
    Param,
    IntParam,
    FloatParam,
    BinaryParam,
    ChoiceParam,
    TupleParam,
)
from randonet.generator.unit import Unit, Factory as _Factory
from randonet.generator.conv import ConvFactory, ConvTransposeFactory
from collections import namedtuple


class LayerNorm(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "LayerNorm", ["normalized_shape", "eps", "elementwise_affine"]
        )
        self.params = self.template_fn(
            normalized_shape=Param(name="normalized_shape", default=None),
            eps=FloatParam(name="eps", default=1e-05),
            elementwise_affine=BinaryParam(
                name="elementwise_affine", default=False, true_prob=0.5
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class LocalResponseNorm(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "LocalResponseNorm", ["size", "alpha", "beta", "k"]
        )
        self.params = self.template_fn(
            size=Param(name="size", default=None),
            alpha=FloatParam(name="alpha", default=0.0001),
            beta=FloatParam(name="beta", default=0.75),
            k=IntParam(name="k", default=1.0),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class CrossMapLRN2d(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("CrossMapLRN2d", ["size", "alpha", "beta", "k"])
        self.params = self.template_fn(
            size=Param(name="size", default=None),
            alpha=FloatParam(name="alpha", default=0.0001),
            beta=FloatParam(name="beta", default=0.75),
            k=IntParam(name="k", default=1),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class GroupNorm(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "GroupNorm", ["num_groups", "num_channels", "eps", "affine"]
        )
        self.params = self.template_fn(
            num_groups=Param(name="num_groups", default=None),
            num_channels=Param(name="num_channels", default=None),
            eps=FloatParam(name="eps", default=1e-05),
            affine=BinaryParam(name="affine", default=False, true_prob=0.5),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v
