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


class CosineSimilarity(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("CosineSimilarity", ["dim", "eps"])
        self.params = self.template_fn(
            dim=IntParam(name="dim", default=1),
            eps=FloatParam(name="eps", default=1e-08),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class PairwiseDistance(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("PairwiseDistance", ["p", "eps", "keepdim"])
        self.params = self.template_fn(
            p=IntParam(name="p", default=2.0),
            eps=FloatParam(name="eps", default=1e-06),
            keepdim=BinaryParam(name="keepdim", default=False, true_prob=0.5),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v
