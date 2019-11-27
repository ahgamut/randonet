{%- extends "base_template.py" %}
{%- block init %}
        {%- for f in layers %}
        self.f{{ loop.index0 }} = nn.{{ f|string }}
        {%- endfor %}
{% endblock %}

{% block forward %}
    def forward(self, *inputs):
        x = inputs[0]
        x = torch.view(x.shape[0],{{ layers[0].in_shape|join(",") }})
        {%- for f in layers %}
        {%- if not loop.first and loop.previtem.out_shape|len != f.in_shape|len %}
        x = torch.view(x.shape[0],{{ f.in_shape|join(",") }})
        {%- endif %}
        x = self.f{{ loop.index0 }}(x)
        {%- endfor %}
        return x
{%- endblock %}
