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


class ReflectionPad1d(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("ReflectionPad1d", ["padding"])
        self.params = self.template_fn(
            padding=TupleParam(
                name="padding", size=1, limits=((0,), (0,)), default=(0,)
            )
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class ReflectionPad2d(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("ReflectionPad2d", ["padding"])
        self.params = self.template_fn(
            padding=TupleParam(
                name="padding", size=2, limits=((0, 0), (0, 0)), default=(0, 0)
            )
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class ReplicationPad1d(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("ReplicationPad1d", ["padding"])
        self.params = self.template_fn(
            padding=TupleParam(
                name="padding", size=1, limits=((0,), (0,)), default=(0,)
            )
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class ReplicationPad2d(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("ReplicationPad2d", ["padding"])
        self.params = self.template_fn(
            padding=TupleParam(
                name="padding", size=2, limits=((0, 0), (0, 0)), default=(0, 0)
            )
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class ReplicationPad3d(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("ReplicationPad3d", ["padding"])
        self.params = self.template_fn(
            padding=TupleParam(
                name="padding", size=3, limits=((0, 0, 0), (0, 0, 0)), default=(0, 0, 0)
            )
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class ZeroPad2d(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("ZeroPad2d", ["padding"])
        self.params = self.template_fn(
            padding=TupleParam(
                name="padding", size=2, limits=((0, 0), (0, 0)), default=(0, 0)
            )
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class ConstantPad1d(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("ConstantPad1d", ["padding", "value"])
        self.params = self.template_fn(
            padding=TupleParam(
                name="padding", size=1, limits=((0,), (0,)), default=(0,)
            ),
            value=FloatParam(name="value", default=0.0),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class ConstantPad2d(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("ConstantPad2d", ["padding", "value"])
        self.params = self.template_fn(
            padding=TupleParam(
                name="padding", size=2, limits=((0, 0), (0, 0)), default=(0, 0)
            ),
            value=FloatParam(name="value", default=0.0),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class ConstantPad3d(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("ConstantPad3d", ["padding", "value"])
        self.params = self.template_fn(
            padding=TupleParam(
                name="padding", size=3, limits=((0, 0, 0), (0, 0, 0)), default=(0, 0, 0)
            ),
            value=FloatParam(name="value", default=0.0),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v
