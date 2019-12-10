{%- extends "base_template.py" %}
{%- block imports %}
{%- if "ResNetStyle" in name %}
from torchvision.models.resnet import BasicBlock
{%- endif %}
{%- endblock%}

{%- block init %}
        {%- for f in layers %}
        {%- if "BasicBlock" in f|string %}
        self.f{{ loop.index0 }} = {{ f | string }}
        {%- else %}
        self.f{{ loop.index0 }} = nn.{{ f|string }}
        {%- endif %}
        {%- endfor %}
{% endblock %}

{% block forward %}
    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],{{ layers[0].in_shape|join(",") }})
        {%- for f in layers %}
        {%- if not loop.first and loop.previtem.out_shape|length != f.in_shape|length %}
        x = x.view(x.shape[0],{{ f.in_shape|join(",") }})
        {%- endif %}
        x = self.f{{ loop.index0 }}(x)
        {%- endfor %}
        return x
{%- endblock %}
