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


class AdaptiveLogSoftmaxWithLoss(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "AdaptiveLogSoftmaxWithLoss",
            ["in_features", "n_classes", "cutoffs", "div_value", "head_bias"],
        )
        self.params = self.template_fn(
            in_features=IntParam(name="in_features", default=1),
            n_classes=Param(name="n_classes", default=None),
            cutoffs=Param(name="cutoffs", default=None),
            div_value=IntParam(name="div_value", default=4.0),
            head_bias=BinaryParam(name="head_bias", default=False, true_prob=0.5),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v
