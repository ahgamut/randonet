# -*- coding: utf-8 -*-
"""
    {{ name.lower() }}.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class {{ name }}(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        {%- block init %}
        {%- endblock %}

    {%- block forward %}
    def forward(self, *inputs):
        pass
    {%- endblock %}
