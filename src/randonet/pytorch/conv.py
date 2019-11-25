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


class Conv1d(ConvFactory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "Conv1d",
            [
                "in_channels",
                "out_channels",
                "kernel_size",
                "stride",
                "padding",
                "dilation",
                "groups",
                "bias",
                "padding_mode",
            ],
        )
        self.params = self.template_fn(
            in_channels=IntParam(name="in_channels", default=1),
            out_channels=IntParam(name="out_channels", default=1),
            kernel_size=TupleParam(
                name="kernel_size", size=1, limits=((1,), (1,)), default=(1,)
            ),
            stride=TupleParam(name="stride", size=1, limits=((1,), (1,)), default=(1,)),
            padding=TupleParam(
                name="padding", size=1, limits=((0,), (0,)), default=(0,)
            ),
            dilation=TupleParam(
                name="dilation", size=1, limits=((1,), (1,)), default=(1,)
            ),
            groups=IntParam(name="groups", default=1),
            bias=BinaryParam(name="bias", default=True, true_prob=0.5),
            padding_mode=ChoiceParam(
                name="padding_mode", choices=("zeros",), cprobs=(1,), default="zeros"
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class Conv2d(ConvFactory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "Conv2d",
            [
                "in_channels",
                "out_channels",
                "kernel_size",
                "stride",
                "padding",
                "dilation",
                "groups",
                "bias",
                "padding_mode",
            ],
        )
        self.params = self.template_fn(
            in_channels=IntParam(name="in_channels", default=1),
            out_channels=IntParam(name="out_channels", default=1),
            kernel_size=TupleParam(
                name="kernel_size", size=2, limits=((1, 1), (1, 1)), default=(1, 1)
            ),
            stride=TupleParam(
                name="stride", size=2, limits=((1, 1), (1, 1)), default=(1, 1)
            ),
            padding=TupleParam(
                name="padding", size=2, limits=((0, 0), (0, 0)), default=(0, 0)
            ),
            dilation=TupleParam(
                name="dilation", size=2, limits=((1, 1), (1, 1)), default=(1, 1)
            ),
            groups=IntParam(name="groups", default=1),
            bias=BinaryParam(name="bias", default=True, true_prob=0.5),
            padding_mode=ChoiceParam(
                name="padding_mode", choices=("zeros",), cprobs=(1,), default="zeros"
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class Conv3d(ConvFactory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "Conv3d",
            [
                "in_channels",
                "out_channels",
                "kernel_size",
                "stride",
                "padding",
                "dilation",
                "groups",
                "bias",
                "padding_mode",
            ],
        )
        self.params = self.template_fn(
            in_channels=IntParam(name="in_channels", default=1),
            out_channels=IntParam(name="out_channels", default=1),
            kernel_size=TupleParam(
                name="kernel_size",
                size=3,
                limits=((1, 1, 1), (1, 1, 1)),
                default=(1, 1, 1),
            ),
            stride=TupleParam(
                name="stride", size=3, limits=((1, 1, 1), (1, 1, 1)), default=(1, 1, 1)
            ),
            padding=TupleParam(
                name="padding", size=3, limits=((0, 0, 0), (0, 0, 0)), default=(0, 0, 0)
            ),
            dilation=TupleParam(
                name="dilation",
                size=3,
                limits=((1, 1, 1), (1, 1, 1)),
                default=(1, 1, 1),
            ),
            groups=IntParam(name="groups", default=1),
            bias=BinaryParam(name="bias", default=True, true_prob=0.5),
            padding_mode=ChoiceParam(
                name="padding_mode", choices=("zeros",), cprobs=(1,), default="zeros"
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class ConvTranspose1d(ConvTransposeFactory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "ConvTranspose1d",
            [
                "in_channels",
                "out_channels",
                "kernel_size",
                "stride",
                "padding",
                "output_padding",
                "groups",
                "bias",
                "dilation",
                "padding_mode",
            ],
        )
        self.params = self.template_fn(
            in_channels=IntParam(name="in_channels", default=1),
            out_channels=IntParam(name="out_channels", default=1),
            kernel_size=TupleParam(
                name="kernel_size", size=1, limits=((1,), (1,)), default=(1,)
            ),
            stride=TupleParam(name="stride", size=1, limits=((1,), (1,)), default=(1,)),
            padding=TupleParam(
                name="padding", size=1, limits=((0,), (0,)), default=(0,)
            ),
            output_padding=TupleParam(
                name="output_padding", size=1, limits=((0,), (0,)), default=(0,)
            ),
            groups=IntParam(name="groups", default=1),
            bias=BinaryParam(name="bias", default=True, true_prob=0.5),
            dilation=TupleParam(
                name="dilation", size=1, limits=((1,), (1,)), default=(1,)
            ),
            padding_mode=ChoiceParam(
                name="padding_mode", choices=("zeros",), cprobs=(1,), default="zeros"
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class ConvTranspose2d(ConvTransposeFactory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "ConvTranspose2d",
            [
                "in_channels",
                "out_channels",
                "kernel_size",
                "stride",
                "padding",
                "output_padding",
                "groups",
                "bias",
                "dilation",
                "padding_mode",
            ],
        )
        self.params = self.template_fn(
            in_channels=IntParam(name="in_channels", default=1),
            out_channels=IntParam(name="out_channels", default=1),
            kernel_size=TupleParam(
                name="kernel_size", size=2, limits=((1, 1), (1, 1)), default=(1, 1)
            ),
            stride=TupleParam(
                name="stride", size=2, limits=((1, 1), (1, 1)), default=(1, 1)
            ),
            padding=TupleParam(
                name="padding", size=2, limits=((0, 0), (0, 0)), default=(0, 0)
            ),
            output_padding=TupleParam(
                name="output_padding", size=2, limits=((0, 0), (0, 0)), default=(0, 0)
            ),
            groups=IntParam(name="groups", default=1),
            bias=BinaryParam(name="bias", default=True, true_prob=0.5),
            dilation=TupleParam(
                name="dilation", size=2, limits=((1, 1), (1, 1)), default=(1, 1)
            ),
            padding_mode=ChoiceParam(
                name="padding_mode", choices=("zeros",), cprobs=(1,), default="zeros"
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class ConvTranspose3d(ConvTransposeFactory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "ConvTranspose3d",
            [
                "in_channels",
                "out_channels",
                "kernel_size",
                "stride",
                "padding",
                "output_padding",
                "groups",
                "bias",
                "dilation",
                "padding_mode",
            ],
        )
        self.params = self.template_fn(
            in_channels=IntParam(name="in_channels", default=1),
            out_channels=IntParam(name="out_channels", default=1),
            kernel_size=TupleParam(
                name="kernel_size",
                size=3,
                limits=((1, 1, 1), (1, 1, 1)),
                default=(1, 1, 1),
            ),
            stride=TupleParam(
                name="stride", size=3, limits=((1, 1, 1), (1, 1, 1)), default=(1, 1, 1)
            ),
            padding=TupleParam(
                name="padding", size=3, limits=((0, 0, 0), (0, 0, 0)), default=(0, 0, 0)
            ),
            output_padding=TupleParam(
                name="output_padding",
                size=3,
                limits=((0, 0, 0), (0, 0, 0)),
                default=(0, 0, 0),
            ),
            groups=IntParam(name="groups", default=1),
            bias=BinaryParam(name="bias", default=True, true_prob=0.5),
            dilation=TupleParam(
                name="dilation",
                size=3,
                limits=((1, 1, 1), (1, 1, 1)),
                default=(1, 1, 1),
            ),
            padding_mode=ChoiceParam(
                name="padding_mode", choices=("zeros",), cprobs=(1,), default="zeros"
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v
