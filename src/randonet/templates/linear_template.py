{%- extends "base_template.py" %}
{%- block init %}
        {%- for f in layers %}
        self.f{{ loop.index0 }} = nn.{{ f|string }}
        {%- endfor %}
{% endblock %}

{% block forward %}
    def forward(self, *inputs):
        x = inputs[0]
        x = torch.view(x[0], {{ layers[0].in_shape|join(",") }})
        {%- for f in layers %}
        x = self.f{{ loop.index0 }}(x)
        {%- endfor %}
        return x
{%- endblock %}