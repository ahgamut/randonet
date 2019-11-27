# -*- coding: utf-8 -*-
"""
    randonet.activation
    ~~~~~~~~~~~~~~~~~~~

    Generation of activation layers, through a choice parameter

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
from randonet.pytorch import (
    Sigmoid,
    Tanh,
    Tanhshrink,
    ReLU,
    ReLU6,
    SELU,
    ELU,
    CELU,
    LeakyReLU,
)
from randonet.generator.param import ChoiceParam


class ActivationParam(ChoiceParam):
    def __init__(self):
        ChoiceParam.__init__(
            self,
            name="Activation",
            choices=[Sigmoid, Tanh, ReLU, ReLU6, SELU],
            cprobs=[i / 5 for i in range(1, 10)],
            is_random=False,
        )
