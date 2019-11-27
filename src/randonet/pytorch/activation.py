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


class Sigmoid(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("Sigmoid", [])
        self.params = self.template_fn()


class Tanh(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("Tanh", [])
        self.params = self.template_fn()


class Softmax2d(ConvFactory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("Softmax2d", [])
        self.params = self.template_fn()


class LogSigmoid(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("LogSigmoid", [])
        self.params = self.template_fn()


class Softsign(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("Softsign", [])
        self.params = self.template_fn()


class Tanhshrink(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("Tanhshrink", [])
        self.params = self.template_fn()


class ReLU(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("ReLU", ["inplace"])
        self.params = self.template_fn(
            inplace=BinaryParam(name="inplace", default=False, true_prob=0.5)
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class ReLU6(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("ReLU6", ["inplace"])
        self.params = self.template_fn(
            inplace=BinaryParam(name="inplace", default=False, true_prob=0.5)
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class Softmax(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("Softmax", ["dim"])
        self.params = self.template_fn(dim=Param(name="dim", default=None))
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class LogSoftmax(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("LogSoftmax", ["dim"])
        self.params = self.template_fn(dim=Param(name="dim", default=None))
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class SELU(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("SELU", ["inplace"])
        self.params = self.template_fn(
            inplace=BinaryParam(name="inplace", default=False, true_prob=0.5)
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class Hardshrink(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("Hardshrink", ["lambd"])
        self.params = self.template_fn(lambd=FloatParam(name="lambd", default=0.5))
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class Softshrink(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("Softshrink", ["lambd"])
        self.params = self.template_fn(lambd=FloatParam(name="lambd", default=0.5))
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class Softmin(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("Softmin", ["dim"])
        self.params = self.template_fn(dim=Param(name="dim", default=None))
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class GLU(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("GLU", ["dim"])
        self.params = self.template_fn(dim=IntParam(name="dim", default=-1))
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class ELU(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("ELU", ["alpha", "inplace"])
        self.params = self.template_fn(
            alpha=FloatParam(name="alpha", default=1.0),
            inplace=BinaryParam(name="inplace", default=False, true_prob=0.5),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class CELU(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("CELU", ["alpha", "inplace"])
        self.params = self.template_fn(
            alpha=FloatParam(name="alpha", default=1.0),
            inplace=BinaryParam(name="inplace", default=False, true_prob=0.5),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class LeakyReLU(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("LeakyReLU", ["negative_slope", "inplace"])
        self.params = self.template_fn(
            negative_slope=FloatParam(name="negative_slope", default=0.01),
            inplace=BinaryParam(name="inplace", default=False, true_prob=0.5),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class Softplus(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("Softplus", ["beta", "threshold"])
        self.params = self.template_fn(
            beta=FloatParam(name="beta", default=1),
            threshold=IntParam(name="threshold", default=20),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class PReLU(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("PReLU", ["num_parameters", "init"])
        self.params = self.template_fn(
            num_parameters=IntParam(name="num_parameters", default=1),
            init=FloatParam(name="init", default=0.25),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class Threshold(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("Threshold", ["threshold", "value", "inplace"])
        self.params = self.template_fn(
            threshold=Param(name="threshold", default=None),
            value=Param(name="value", default=None),
            inplace=BinaryParam(name="inplace", default=False, true_prob=0.5),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class RReLU(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("RReLU", ["lower", "upper", "inplace"])
        self.params = self.template_fn(
            lower=FloatParam(name="lower", default=0.125),
            upper=FloatParam(name="upper", default=0.3333333333333333),
            inplace=BinaryParam(name="inplace", default=False, true_prob=0.5),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class Hardtanh(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "Hardtanh", ["min_val", "max_val", "inplace", "min_value", "max_value"]
        )
        self.params = self.template_fn(
            min_val=IntParam(name="min_val", default=-1.0),
            max_val=IntParam(name="max_val", default=1.0),
            inplace=BinaryParam(name="inplace", default=False, true_prob=0.5),
            min_value=Param(name="min_value", default=None),
            max_value=Param(name="max_value", default=None),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class MultiheadAttention(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "MultiheadAttention",
            [
                "embed_dim",
                "num_heads",
                "dropout",
                "bias",
                "add_bias_kv",
                "add_zero_attn",
                "kdim",
                "vdim",
            ],
        )
        self.params = self.template_fn(
            embed_dim=Param(name="embed_dim", default=None),
            num_heads=Param(name="num_heads", default=None),
            dropout=IntParam(name="dropout", default=0.0),
            bias=BinaryParam(name="bias", default=True, true_prob=0.5),
            add_bias_kv=BinaryParam(name="add_bias_kv", default=False, true_prob=0.5),
            add_zero_attn=BinaryParam(
                name="add_zero_attn", default=False, true_prob=0.5
            ),
            kdim=Param(name="kdim", default=None),
            vdim=Param(name="vdim", default=None),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v
