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


class Identity(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("Identity", [])
        self.params = self.template_fn()


class Linear(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("Linear", ["in_features", "out_features", "bias"])
        self.params = self.template_fn(
            in_features=IntParam(name="in_features", default=1),
            out_features=IntParam(name="out_features", default=1),
            bias=BinaryParam(name="bias", default=True, true_prob=0.5),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v

    def _fix_inshape(self, _in_shape):
        if len(_in_shape) != 1:
            num = 1
            for x in _in_shape:
                num = num * x
            return [num]
        return _in_shape

    def _get_outshape(self, fn, in_shape):
        return [fn.out_features]

    def _lock_kernel(self, fn, in_shape, out_shape):
        return fn._replace(in_features=in_shape[0], out_features=out_shape[0])

    def __call__(self, _in_shape, _out_shape=None):
        in_shape = self._fix_inshape(_in_shape)
        self.in_features.val = in_shape[0]
        fn = self._render()
        if _out_shape is not None:
            out_shape = _out_shape
            fn = self._lock_kernel(fn, in_shape, out_shape)
        else:
            out_shape = self._get_outshape(fn, in_shape)
        return Unit(fn, in_shape, out_shape)


class Bilinear(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "Bilinear", ["in1_features", "in2_features", "out_features", "bias"]
        )
        self.params = self.template_fn(
            in1_features=IntParam(name="in1_features", default=1),
            in2_features=IntParam(name="in2_features", default=1),
            out_features=IntParam(name="out_features", default=1),
            bias=BinaryParam(name="bias", default=True, true_prob=0.5),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v
