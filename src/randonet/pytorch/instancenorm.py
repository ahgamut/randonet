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


class InstanceNorm1d(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "InstanceNorm1d",
            ["num_features", "eps", "momentum", "affine", "track_running_stats"],
        )
        self.params = self.template_fn(
            num_features=Param(name="num_features", default=None),
            eps=FloatParam(name="eps", default=1e-05),
            momentum=FloatParam(name="momentum", default=0.1),
            affine=BinaryParam(name="affine", default=False, true_prob=0.5),
            track_running_stats=BinaryParam(
                name="track_running_stats", default=False, true_prob=0.5
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class InstanceNorm2d(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "InstanceNorm2d",
            ["num_features", "eps", "momentum", "affine", "track_running_stats"],
        )
        self.params = self.template_fn(
            num_features=Param(name="num_features", default=None),
            eps=FloatParam(name="eps", default=1e-05),
            momentum=FloatParam(name="momentum", default=0.1),
            affine=BinaryParam(name="affine", default=False, true_prob=0.5),
            track_running_stats=BinaryParam(
                name="track_running_stats", default=False, true_prob=0.5
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class InstanceNorm3d(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "InstanceNorm3d",
            ["num_features", "eps", "momentum", "affine", "track_running_stats"],
        )
        self.params = self.template_fn(
            num_features=Param(name="num_features", default=None),
            eps=FloatParam(name="eps", default=1e-05),
            momentum=FloatParam(name="momentum", default=0.1),
            affine=BinaryParam(name="affine", default=False, true_prob=0.5),
            track_running_stats=BinaryParam(
                name="track_running_stats", default=False, true_prob=0.5
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v
