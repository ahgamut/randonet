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


class AdaptiveAvgPool1d(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("AdaptiveAvgPool1d", ["output_size"])
        self.params = self.template_fn(
            output_size=TupleParam(name="output_size", size=1, default=(1,))
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class AdaptiveAvgPool2d(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("AdaptiveAvgPool2d", ["output_size"])
        self.params = self.template_fn(
            output_size=TupleParam(name="output_size", size=2, default=(1, 1))
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class AdaptiveAvgPool3d(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("AdaptiveAvgPool3d", ["output_size"])
        self.params = self.template_fn(
            output_size=TupleParam(name="output_size", size=3, default=(1, 1, 1))
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class AdaptiveMaxPool1d(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "AdaptiveMaxPool1d", ["output_size", "return_indices"]
        )
        self.params = self.template_fn(
            output_size=TupleParam(name="output_size", size=1, default=(1,)),
            return_indices=BinaryParam(
                name="return_indices", default=False, true_prob=0.5
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class AdaptiveMaxPool2d(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "AdaptiveMaxPool2d", ["output_size", "return_indices"]
        )
        self.params = self.template_fn(
            output_size=TupleParam(name="output_size", size=2, default=(1, 1)),
            return_indices=BinaryParam(
                name="return_indices", default=False, true_prob=0.5
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class AdaptiveMaxPool3d(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "AdaptiveMaxPool3d", ["output_size", "return_indices"]
        )
        self.params = self.template_fn(
            output_size=TupleParam(name="output_size", size=3, default=(1, 1, 1)),
            return_indices=BinaryParam(
                name="return_indices", default=False, true_prob=0.5
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class MaxUnpool1d(ConvFactory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "MaxUnpool1d", ["kernel_size", "stride", "padding"]
        )
        self.params = self.template_fn(
            kernel_size=TupleParam(
                name="kernel_size", size=1, limits=((1,), (1,)), default=(1,)
            ),
            stride=TupleParam(name="stride", size=1, limits=((1,), (1,)), default=(1,)),
            padding=TupleParam(
                name="padding", size=1, limits=((0,), (0,)), default=(0,)
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class MaxUnpool2d(ConvFactory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "MaxUnpool2d", ["kernel_size", "stride", "padding"]
        )
        self.params = self.template_fn(
            kernel_size=TupleParam(
                name="kernel_size", size=2, limits=((1, 1), (1, 1)), default=(1, 1)
            ),
            stride=TupleParam(
                name="stride", size=2, limits=((1, 1), (1, 1)), default=(1, 1)
            ),
            padding=TupleParam(
                name="padding", size=2, limits=((0, 0), (0, 0)), default=(0, 0)
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class MaxUnpool3d(ConvFactory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "MaxUnpool3d", ["kernel_size", "stride", "padding"]
        )
        self.params = self.template_fn(
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
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class LPPool1d(ConvFactory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "LPPool1d", ["norm_type", "kernel_size", "stride", "ceil_mode"]
        )
        self.params = self.template_fn(
            norm_type=Param(name="norm_type", default=None),
            kernel_size=TupleParam(
                name="kernel_size", size=1, limits=((1,), (1,)), default=(1,)
            ),
            stride=TupleParam(name="stride", size=1, limits=((1,), (1,)), default=(1,)),
            ceil_mode=BinaryParam(name="ceil_mode", default=False, true_prob=0.5),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class LPPool2d(ConvFactory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "LPPool2d", ["norm_type", "kernel_size", "stride", "ceil_mode"]
        )
        self.params = self.template_fn(
            norm_type=Param(name="norm_type", default=None),
            kernel_size=TupleParam(
                name="kernel_size", size=2, limits=((1, 1), (1, 1)), default=(1, 1)
            ),
            stride=TupleParam(
                name="stride", size=2, limits=((1, 1), (1, 1)), default=(1, 1)
            ),
            ceil_mode=BinaryParam(name="ceil_mode", default=False, true_prob=0.5),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class AvgPool1d(ConvFactory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "AvgPool1d",
            ["kernel_size", "stride", "padding", "ceil_mode", "count_include_pad"],
        )
        self.params = self.template_fn(
            kernel_size=TupleParam(
                name="kernel_size", size=1, limits=((1,), (1,)), default=(1,)
            ),
            stride=TupleParam(name="stride", size=1, limits=((1,), (1,)), default=(1,)),
            padding=TupleParam(
                name="padding", size=1, limits=((0,), (0,)), default=(0,)
            ),
            ceil_mode=BinaryParam(name="ceil_mode", default=False, true_prob=0.5),
            count_include_pad=BinaryParam(
                name="count_include_pad", default=True, true_prob=0.5
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class FractionalMaxPool2d(ConvFactory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "FractionalMaxPool2d",
            [
                "kernel_size",
                "output_size",
                "output_ratio",
                "return_indices",
                "_random_samples",
            ],
        )
        self.params = self.template_fn(
            kernel_size=TupleParam(
                name="kernel_size", size=2, limits=((1, 1), (1, 1)), default=(1, 1)
            ),
            output_size=TupleParam(name="output_size", size=2, default=(1, 1)),
            output_ratio=Param(name="output_ratio", default=None),
            return_indices=BinaryParam(
                name="return_indices", default=False, true_prob=0.5
            ),
            _random_samples=Param(name="_random_samples", default=None),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class FractionalMaxPool3d(ConvFactory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "FractionalMaxPool3d",
            [
                "kernel_size",
                "output_size",
                "output_ratio",
                "return_indices",
                "_random_samples",
            ],
        )
        self.params = self.template_fn(
            kernel_size=TupleParam(
                name="kernel_size",
                size=3,
                limits=((1, 1, 1), (1, 1, 1)),
                default=(1, 1, 1),
            ),
            output_size=TupleParam(name="output_size", size=3, default=(1, 1, 1)),
            output_ratio=Param(name="output_ratio", default=None),
            return_indices=BinaryParam(
                name="return_indices", default=False, true_prob=0.5
            ),
            _random_samples=Param(name="_random_samples", default=None),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class AvgPool2d(ConvFactory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "AvgPool2d",
            [
                "kernel_size",
                "stride",
                "padding",
                "ceil_mode",
                "count_include_pad",
                "divisor_override",
            ],
        )
        self.params = self.template_fn(
            kernel_size=TupleParam(
                name="kernel_size", size=2, limits=((1, 1), (1, 1)), default=(1, 1)
            ),
            stride=TupleParam(
                name="stride", size=2, limits=((1, 1), (1, 1)), default=(1, 1)
            ),
            padding=TupleParam(
                name="padding", size=2, limits=((0, 0), (0, 0)), default=(0, 0)
            ),
            ceil_mode=BinaryParam(name="ceil_mode", default=False, true_prob=0.5),
            count_include_pad=BinaryParam(
                name="count_include_pad", default=True, true_prob=0.5
            ),
            divisor_override=Param(name="divisor_override", default=None),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class AvgPool3d(ConvFactory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "AvgPool3d",
            [
                "kernel_size",
                "stride",
                "padding",
                "ceil_mode",
                "count_include_pad",
                "divisor_override",
            ],
        )
        self.params = self.template_fn(
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
            ceil_mode=BinaryParam(name="ceil_mode", default=False, true_prob=0.5),
            count_include_pad=BinaryParam(
                name="count_include_pad", default=True, true_prob=0.5
            ),
            divisor_override=Param(name="divisor_override", default=None),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class MaxPool1d(ConvFactory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "MaxPool1d",
            [
                "kernel_size",
                "stride",
                "padding",
                "dilation",
                "return_indices",
                "ceil_mode",
            ],
        )
        self.params = self.template_fn(
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
            return_indices=BinaryParam(
                name="return_indices", default=False, true_prob=0.5
            ),
            ceil_mode=BinaryParam(name="ceil_mode", default=False, true_prob=0.5),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class MaxPool2d(ConvFactory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "MaxPool2d",
            [
                "kernel_size",
                "stride",
                "padding",
                "dilation",
                "return_indices",
                "ceil_mode",
            ],
        )
        self.params = self.template_fn(
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
            return_indices=BinaryParam(
                name="return_indices", default=False, true_prob=0.5
            ),
            ceil_mode=BinaryParam(name="ceil_mode", default=False, true_prob=0.5),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class MaxPool3d(ConvFactory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "MaxPool3d",
            [
                "kernel_size",
                "stride",
                "padding",
                "dilation",
                "return_indices",
                "ceil_mode",
            ],
        )
        self.params = self.template_fn(
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
            return_indices=BinaryParam(
                name="return_indices", default=False, true_prob=0.5
            ),
            ceil_mode=BinaryParam(name="ceil_mode", default=False, true_prob=0.5),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v
