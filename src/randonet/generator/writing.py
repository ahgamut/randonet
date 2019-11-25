# -*- coding: utf-8 -*-
"""
    randonet.generator
    ~~~~~~~~~~~~~~~~~~

    Converts neural net template into a file to load
    and use in computations

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
from jinja2 import Environment, select_autoescape, PackageLoader


def write_template(info_dict, file_path):
    env = Environment(loader=PackageLoader("randonet", "templates"))
    temp = env.get_template("linear_template.py")
    with open(file_path, "w+") as f:
        f.write(temp.render(**info_dict))


def print_sample():
    env = Environment(loader=PackageLoader("randonet", "templates"))
    temp = env.get_template("linear_template.py")
    return temp.render(name="Net_Zero", start=60, stop=1, features=[50, 40, 30, 20, 10])
