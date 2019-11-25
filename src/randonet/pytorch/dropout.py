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


class Dropout(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("Dropout", ["p", "inplace"])
        self.params = self.template_fn(
            p=FloatParam(name="p", default=0.5),
            inplace=BinaryParam(name="inplace", default=False, true_prob=0.5),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class Dropout2d(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("Dropout2d", ["p", "inplace"])
        self.params = self.template_fn(
            p=FloatParam(name="p", default=0.5),
            inplace=BinaryParam(name="inplace", default=False, true_prob=0.5),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class Dropout3d(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("Dropout3d", ["p", "inplace"])
        self.params = self.template_fn(
            p=FloatParam(name="p", default=0.5),
            inplace=BinaryParam(name="inplace", default=False, true_prob=0.5),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class AlphaDropout(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("AlphaDropout", ["p", "inplace"])
        self.params = self.template_fn(
            p=FloatParam(name="p", default=0.5),
            inplace=BinaryParam(name="inplace", default=False, true_prob=0.5),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class FeatureAlphaDropout(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("FeatureAlphaDropout", ["p", "inplace"])
        self.params = self.template_fn(
            p=FloatParam(name="p", default=0.5),
            inplace=BinaryParam(name="inplace", default=False, true_prob=0.5),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v
