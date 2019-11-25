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


class Embedding(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "Embedding",
            [
                "num_embeddings",
                "embedding_dim",
                "padding_idx",
                "max_norm",
                "norm_type",
                "scale_grad_by_freq",
                "sparse",
                "_weight",
            ],
        )
        self.params = self.template_fn(
            num_embeddings=IntParam(name="num_embeddings", default=1),
            embedding_dim=IntParam(name="embedding_dim", default=1),
            padding_idx=Param(name="padding_idx", default=None),
            max_norm=Param(name="max_norm", default=None),
            norm_type=IntParam(name="norm_type", default=2.0),
            scale_grad_by_freq=BinaryParam(
                name="scale_grad_by_freq", default=False, true_prob=0.5
            ),
            sparse=BinaryParam(name="sparse", default=False, true_prob=0.5),
            _weight=Param(name="_weight", default=None),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class EmbeddingBag(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "EmbeddingBag",
            [
                "num_embeddings",
                "embedding_dim",
                "max_norm",
                "norm_type",
                "scale_grad_by_freq",
                "mode",
                "sparse",
                "_weight",
            ],
        )
        self.params = self.template_fn(
            num_embeddings=IntParam(name="num_embeddings", default=1),
            embedding_dim=IntParam(name="embedding_dim", default=1),
            max_norm=Param(name="max_norm", default=None),
            norm_type=IntParam(name="norm_type", default=2.0),
            scale_grad_by_freq=BinaryParam(
                name="scale_grad_by_freq", default=False, true_prob=0.5
            ),
            mode=ChoiceParam(
                name="mode", choices=("mean",), cprobs=(1,), default="mean"
            ),
            sparse=BinaryParam(name="sparse", default=False, true_prob=0.5),
            _weight=Param(name="_weight", default=None),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v
