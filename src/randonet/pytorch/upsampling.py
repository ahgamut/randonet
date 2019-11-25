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


class UpsamplingNearest2d(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("UpsamplingNearest2d", ["size", "scale_factor"])
        self.params = self.template_fn(
            size=Param(name="size", default=None),
            scale_factor=IntParam(name="scale_factor", default=1),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class UpsamplingBilinear2d(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("UpsamplingBilinear2d", ["size", "scale_factor"])
        self.params = self.template_fn(
            size=Param(name="size", default=None),
            scale_factor=IntParam(name="scale_factor", default=1),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class Upsample(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "Upsample", ["size", "scale_factor", "mode", "align_corners"]
        )
        self.params = self.template_fn(
            size=Param(name="size", default=None),
            scale_factor=IntParam(name="scale_factor", default=1),
            mode=ChoiceParam(
                name="mode", choices=("nearest",), cprobs=(1,), default="nearest"
            ),
            align_corners=Param(name="align_corners", default=None),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v
