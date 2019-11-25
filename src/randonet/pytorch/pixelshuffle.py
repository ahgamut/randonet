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


class PixelShuffle(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("PixelShuffle", ["upscale_factor"])
        self.params = self.template_fn(
            upscale_factor=IntParam(name="upscale_factor", default=1)
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v
