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


class Flatten(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("Flatten", ["start_dim", "end_dim"])
        self.params = self.template_fn(
            start_dim=IntParam(name="start_dim", default=1),
            end_dim=IntParam(name="end_dim", default=-1),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v
