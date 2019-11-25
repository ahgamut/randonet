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


class L1Loss(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("L1Loss", ["size_average", "reduce", "reduction"])
        self.params = self.template_fn(
            size_average=Param(name="size_average", default=None),
            reduce=Param(name="reduce", default=None),
            reduction=ChoiceParam(
                name="reduction", choices=("mean",), cprobs=(1,), default="mean"
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class KLDivLoss(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "KLDivLoss", ["size_average", "reduce", "reduction"]
        )
        self.params = self.template_fn(
            size_average=Param(name="size_average", default=None),
            reduce=Param(name="reduce", default=None),
            reduction=ChoiceParam(
                name="reduction", choices=("mean",), cprobs=(1,), default="mean"
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class MSELoss(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "MSELoss", ["size_average", "reduce", "reduction"]
        )
        self.params = self.template_fn(
            size_average=Param(name="size_average", default=None),
            reduce=Param(name="reduce", default=None),
            reduction=ChoiceParam(
                name="reduction", choices=("mean",), cprobs=(1,), default="mean"
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class CTCLoss(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "CTCLoss", ["blank", "reduction", "zero_infinity"]
        )
        self.params = self.template_fn(
            blank=IntParam(name="blank", default=0),
            reduction=ChoiceParam(
                name="reduction", choices=("mean",), cprobs=(1,), default="mean"
            ),
            zero_infinity=BinaryParam(
                name="zero_infinity", default=False, true_prob=0.5
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class MultiLabelMarginLoss(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "MultiLabelMarginLoss", ["size_average", "reduce", "reduction"]
        )
        self.params = self.template_fn(
            size_average=Param(name="size_average", default=None),
            reduce=Param(name="reduce", default=None),
            reduction=ChoiceParam(
                name="reduction", choices=("mean",), cprobs=(1,), default="mean"
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class SmoothL1Loss(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "SmoothL1Loss", ["size_average", "reduce", "reduction"]
        )
        self.params = self.template_fn(
            size_average=Param(name="size_average", default=None),
            reduce=Param(name="reduce", default=None),
            reduction=ChoiceParam(
                name="reduction", choices=("mean",), cprobs=(1,), default="mean"
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class SoftMarginLoss(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "SoftMarginLoss", ["size_average", "reduce", "reduction"]
        )
        self.params = self.template_fn(
            size_average=Param(name="size_average", default=None),
            reduce=Param(name="reduce", default=None),
            reduction=ChoiceParam(
                name="reduction", choices=("mean",), cprobs=(1,), default="mean"
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class BCELoss(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "BCELoss", ["weight", "size_average", "reduce", "reduction"]
        )
        self.params = self.template_fn(
            weight=Param(name="weight", default=None),
            size_average=Param(name="size_average", default=None),
            reduce=Param(name="reduce", default=None),
            reduction=ChoiceParam(
                name="reduction", choices=("mean",), cprobs=(1,), default="mean"
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class CosineEmbeddingLoss(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "CosineEmbeddingLoss", ["margin", "size_average", "reduce", "reduction"]
        )
        self.params = self.template_fn(
            margin=IntParam(name="margin", default=0.0),
            size_average=Param(name="size_average", default=None),
            reduce=Param(name="reduce", default=None),
            reduction=ChoiceParam(
                name="reduction", choices=("mean",), cprobs=(1,), default="mean"
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class HingeEmbeddingLoss(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "HingeEmbeddingLoss", ["margin", "size_average", "reduce", "reduction"]
        )
        self.params = self.template_fn(
            margin=IntParam(name="margin", default=1.0),
            size_average=Param(name="size_average", default=None),
            reduce=Param(name="reduce", default=None),
            reduction=ChoiceParam(
                name="reduction", choices=("mean",), cprobs=(1,), default="mean"
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class MarginRankingLoss(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "MarginRankingLoss", ["margin", "size_average", "reduce", "reduction"]
        )
        self.params = self.template_fn(
            margin=IntParam(name="margin", default=0.0),
            size_average=Param(name="size_average", default=None),
            reduce=Param(name="reduce", default=None),
            reduction=ChoiceParam(
                name="reduction", choices=("mean",), cprobs=(1,), default="mean"
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class MultiLabelSoftMarginLoss(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "MultiLabelSoftMarginLoss",
            ["weight", "size_average", "reduce", "reduction"],
        )
        self.params = self.template_fn(
            weight=Param(name="weight", default=None),
            size_average=Param(name="size_average", default=None),
            reduce=Param(name="reduce", default=None),
            reduction=ChoiceParam(
                name="reduction", choices=("mean",), cprobs=(1,), default="mean"
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class NLLLoss(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "NLLLoss", ["weight", "size_average", "ignore_index", "reduce", "reduction"]
        )
        self.params = self.template_fn(
            weight=Param(name="weight", default=None),
            size_average=Param(name="size_average", default=None),
            ignore_index=IntParam(name="ignore_index", default=-100),
            reduce=Param(name="reduce", default=None),
            reduction=ChoiceParam(
                name="reduction", choices=("mean",), cprobs=(1,), default="mean"
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class BCEWithLogitsLoss(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "BCEWithLogitsLoss",
            ["weight", "size_average", "reduce", "reduction", "pos_weight"],
        )
        self.params = self.template_fn(
            weight=Param(name="weight", default=None),
            size_average=Param(name="size_average", default=None),
            reduce=Param(name="reduce", default=None),
            reduction=ChoiceParam(
                name="reduction", choices=("mean",), cprobs=(1,), default="mean"
            ),
            pos_weight=Param(name="pos_weight", default=None),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class NLLLoss2d(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "NLLLoss2d",
            ["weight", "size_average", "ignore_index", "reduce", "reduction"],
        )
        self.params = self.template_fn(
            weight=Param(name="weight", default=None),
            size_average=Param(name="size_average", default=None),
            ignore_index=IntParam(name="ignore_index", default=-100),
            reduce=Param(name="reduce", default=None),
            reduction=ChoiceParam(
                name="reduction", choices=("mean",), cprobs=(1,), default="mean"
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class CrossEntropyLoss(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "CrossEntropyLoss",
            ["weight", "size_average", "ignore_index", "reduce", "reduction"],
        )
        self.params = self.template_fn(
            weight=Param(name="weight", default=None),
            size_average=Param(name="size_average", default=None),
            ignore_index=IntParam(name="ignore_index", default=-100),
            reduce=Param(name="reduce", default=None),
            reduction=ChoiceParam(
                name="reduction", choices=("mean",), cprobs=(1,), default="mean"
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class MultiMarginLoss(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "MultiMarginLoss",
            ["p", "margin", "weight", "size_average", "reduce", "reduction"],
        )
        self.params = self.template_fn(
            p=IntParam(name="p", default=1),
            margin=IntParam(name="margin", default=1.0),
            weight=Param(name="weight", default=None),
            size_average=Param(name="size_average", default=None),
            reduce=Param(name="reduce", default=None),
            reduction=ChoiceParam(
                name="reduction", choices=("mean",), cprobs=(1,), default="mean"
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class PoissonNLLLoss(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "PoissonNLLLoss",
            ["log_input", "full", "size_average", "eps", "reduce", "reduction"],
        )
        self.params = self.template_fn(
            log_input=BinaryParam(name="log_input", default=True, true_prob=0.5),
            full=BinaryParam(name="full", default=False, true_prob=0.5),
            size_average=Param(name="size_average", default=None),
            eps=FloatParam(name="eps", default=1e-08),
            reduce=Param(name="reduce", default=None),
            reduction=ChoiceParam(
                name="reduction", choices=("mean",), cprobs=(1,), default="mean"
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class TripletMarginLoss(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "TripletMarginLoss",
            ["margin", "p", "eps", "swap", "size_average", "reduce", "reduction"],
        )
        self.params = self.template_fn(
            margin=IntParam(name="margin", default=1.0),
            p=IntParam(name="p", default=2.0),
            eps=FloatParam(name="eps", default=1e-06),
            swap=BinaryParam(name="swap", default=False, true_prob=0.5),
            size_average=Param(name="size_average", default=None),
            reduce=Param(name="reduce", default=None),
            reduction=ChoiceParam(
                name="reduction", choices=("mean",), cprobs=(1,), default="mean"
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v
