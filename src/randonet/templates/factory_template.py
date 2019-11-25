{%- set factype = "_Factory" %}
{%- if ("1d" in name or "2d" in name or "3d" in name) and "kernel_size" in param_names %}
    {%- if "Transpose" in name %}
    {%- set factype = "ConvTransposeFactory" %}
    {%- else %}
    {%- set factype = "ConvFactory" %}
    {%- endif %}
{%- endif %}
class {{ name }}({{ factype }}):
    def __init__(self, **kwargs):
        _Factory.__init__(self)
        self.template_fn = namedtuple("{{ name }}", {{ param_names }})
        self.params = self.template_fn(
        {%- for n, val in params.items() %}
            {%- if val|string in ["True", "False"] %}
            {{ n }}=BinaryParam(name="{{ n }}", default={{ val }}, true_prob=0.5),
            {%- elif val is number and val == val|int %}
            {{ n }}=IntParam(name="{{ n }}", default={{ val }}),
            {%- elif val is number and val != val|int %}
            {{ n }}=FloatParam(name="{{ n }}", default={{ val }}),
            {%- elif val is string %}
            {{ n }}=ChoiceParam(name="{{ n }}", choices=({{ val }},), cprobs=(1,), default={{ val }}),
            {%- elif val is iterable %}
            {{ n }}=TupleParam(name="{{ n }}", size={{ val|length }}, limits={{ (val, val,) }}, default={{ val }}),
            {%- else %}
            {{ n }}=Param(name="{{ n }}", default={{ val }}),
            {%- endif %}
        {%- endfor %}
        )
        {%- if param_names|length > 0 %}
        for k,v in kwargs.items():
            getattr(self.params, k).val = v
        {%- endif %}
