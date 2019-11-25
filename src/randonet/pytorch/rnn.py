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


class RNN(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "RNN",
            [
                "input_size",
                "hidden_size",
                "num_layers",
                "bias",
                "batch_first",
                "dropout",
                "bidirectional",
            ],
        )
        self.params = self.template_fn(
            input_size=IntParam(name="input_size", default=1),
            hidden_size=Param(name="hidden_size", default=None),
            num_layers=IntParam(name="num_layers", default=1),
            bias=BinaryParam(name="bias", default=True, true_prob=0.5),
            batch_first=BinaryParam(name="batch_first", default=False, true_prob=0.5),
            dropout=IntParam(name="dropout", default=0.0),
            bidirectional=BinaryParam(
                name="bidirectional", default=False, true_prob=0.5
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class LSTM(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "LSTM",
            [
                "input_size",
                "hidden_size",
                "num_layers",
                "bias",
                "batch_first",
                "dropout",
                "bidirectional",
            ],
        )
        self.params = self.template_fn(
            input_size=IntParam(name="input_size", default=1),
            hidden_size=Param(name="hidden_size", default=None),
            num_layers=IntParam(name="num_layers", default=1),
            bias=BinaryParam(name="bias", default=True, true_prob=0.5),
            batch_first=BinaryParam(name="batch_first", default=False, true_prob=0.5),
            dropout=IntParam(name="dropout", default=0.0),
            bidirectional=BinaryParam(
                name="bidirectional", default=False, true_prob=0.5
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class GRU(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "GRU",
            [
                "input_size",
                "hidden_size",
                "num_layers",
                "bias",
                "batch_first",
                "dropout",
                "bidirectional",
            ],
        )
        self.params = self.template_fn(
            input_size=IntParam(name="input_size", default=1),
            hidden_size=Param(name="hidden_size", default=None),
            num_layers=IntParam(name="num_layers", default=1),
            bias=BinaryParam(name="bias", default=True, true_prob=0.5),
            batch_first=BinaryParam(name="batch_first", default=False, true_prob=0.5),
            dropout=IntParam(name="dropout", default=0.0),
            bidirectional=BinaryParam(
                name="bidirectional", default=False, true_prob=0.5
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class LSTMCell(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("LSTMCell", ["input_size", "hidden_size", "bias"])
        self.params = self.template_fn(
            input_size=IntParam(name="input_size", default=1),
            hidden_size=Param(name="hidden_size", default=None),
            bias=BinaryParam(name="bias", default=True, true_prob=0.5),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class GRUCell(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("GRUCell", ["input_size", "hidden_size", "bias"])
        self.params = self.template_fn(
            input_size=IntParam(name="input_size", default=1),
            hidden_size=Param(name="hidden_size", default=None),
            bias=BinaryParam(name="bias", default=True, true_prob=0.5),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class RNNCellBase(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "RNNCellBase", ["input_size", "hidden_size", "bias", "num_chunks"]
        )
        self.params = self.template_fn(
            input_size=IntParam(name="input_size", default=1),
            hidden_size=Param(name="hidden_size", default=None),
            bias=Param(name="bias", default=None),
            num_chunks=Param(name="num_chunks", default=None),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class RNNCell(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "RNNCell", ["input_size", "hidden_size", "bias", "nonlinearity"]
        )
        self.params = self.template_fn(
            input_size=IntParam(name="input_size", default=1),
            hidden_size=Param(name="hidden_size", default=None),
            bias=BinaryParam(name="bias", default=True, true_prob=0.5),
            nonlinearity=ChoiceParam(
                name="nonlinearity", choices=("tanh",), cprobs=(1,), default="tanh"
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v


class RNNBase(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple(
            "RNNBase",
            [
                "mode",
                "input_size",
                "hidden_size",
                "num_layers",
                "bias",
                "batch_first",
                "dropout",
                "bidirectional",
            ],
        )
        self.params = self.template_fn(
            mode=Param(name="mode", default=None),
            input_size=IntParam(name="input_size", default=1),
            hidden_size=Param(name="hidden_size", default=None),
            num_layers=IntParam(name="num_layers", default=1),
            bias=BinaryParam(name="bias", default=True, true_prob=0.5),
            batch_first=BinaryParam(name="batch_first", default=False, true_prob=0.5),
            dropout=IntParam(name="dropout", default=0.0),
            bidirectional=BinaryParam(
                name="bidirectional", default=False, true_prob=0.5
            ),
        )
        for k, v in kwargs.items():
            getattr(self.params, k).val = v
