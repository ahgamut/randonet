
from randonet.generator.param import Param, IntParam, FloatParam, BinaryParam, ChoiceParam, TupleParam
from randonet.generator.unit import Unit, Factory as _Factory
from randonet.generator.conv import ConvFactory, ConvTransposeFactory
from collections import namedtuple


class TransformerEncoder(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("TransformerEncoder", ['encoder_layer', 'num_layers', 'norm'])
        self.params = self.template_fn(
            encoder_layer=Param(name="encoder_layer", default=None),
            num_layers=Param(name="num_layers", default=None),
            norm=Param(name="norm", default=None),
        )
        for k,v in kwargs.items():
            getattr(self.params, k).val = v


class TransformerDecoder(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("TransformerDecoder", ['decoder_layer', 'num_layers', 'norm'])
        self.params = self.template_fn(
            decoder_layer=Param(name="decoder_layer", default=None),
            num_layers=Param(name="num_layers", default=None),
            norm=Param(name="norm", default=None),
        )
        for k,v in kwargs.items():
            getattr(self.params, k).val = v


class TransformerEncoderLayer(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("TransformerEncoderLayer", ['d_model', 'nhead', 'dim_feedforward', 'dropout', 'activation'])
        self.params = self.template_fn(
            d_model=Param(name="d_model", default=None),
            nhead=Param(name="nhead", default=None),
            dim_feedforward=IntParam(name="dim_feedforward", default=2048),
            dropout=FloatParam(name="dropout", default=0.1),
            activation=ChoiceParam(name="activation", choices=("relu",), cprobs=(1,), default="relu"),
        )
        for k,v in kwargs.items():
            getattr(self.params, k).val = v


class TransformerDecoderLayer(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("TransformerDecoderLayer", ['d_model', 'nhead', 'dim_feedforward', 'dropout', 'activation'])
        self.params = self.template_fn(
            d_model=Param(name="d_model", default=None),
            nhead=Param(name="nhead", default=None),
            dim_feedforward=IntParam(name="dim_feedforward", default=2048),
            dropout=FloatParam(name="dropout", default=0.1),
            activation=ChoiceParam(name="activation", choices=("relu",), cprobs=(1,), default="relu"),
        )
        for k,v in kwargs.items():
            getattr(self.params, k).val = v


class Transformer(_Factory):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("Transformer", ['d_model', 'nhead', 'num_encoder_layers', 'num_decoder_layers', 'dim_feedforward', 'dropout', 'activation', 'custom_encoder', 'custom_decoder'])
        self.params = self.template_fn(
            d_model=IntParam(name="d_model", default=512),
            nhead=IntParam(name="nhead", default=8),
            num_encoder_layers=IntParam(name="num_encoder_layers", default=6),
            num_decoder_layers=IntParam(name="num_decoder_layers", default=6),
            dim_feedforward=IntParam(name="dim_feedforward", default=2048),
            dropout=FloatParam(name="dropout", default=0.1),
            activation=ChoiceParam(name="activation", choices=("relu",), cprobs=(1,), default="relu"),
            custom_encoder=Param(name="custom_encoder", default=None),
            custom_decoder=Param(name="custom_decoder", default=None),
        )
        for k,v in kwargs.items():
            getattr(self.params, k).val = v

