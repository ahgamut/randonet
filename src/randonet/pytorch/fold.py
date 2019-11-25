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


class Unfold(ConvFactory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "Unfold", ["kernel_size", "dilation", "padding", "stride"]
        )
        self.params = self.template_fn(
            kernel_size=Param(name="kernel_size", default=None),
            dilation=IntParam(name="dilation", default=1),
            padding=IntParam(name="padding", default=0),
            stride=IntParam(name="stride", default=1),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class Fold(ConvFactory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "Fold", ["output_size", "kernel_size", "dilation", "padding", "stride"]
        )
        self.params = self.template_fn(
            output_size=IntParam(name="output_size", default=1),
            kernel_size=Param(name="kernel_size", default=None),
            dilation=IntParam(name="dilation", default=1),
            padding=IntParam(name="padding", default=0),
            stride=IntParam(name="stride", default=1),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v
