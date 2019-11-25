import torch
import sys, os
import inspect
from jinja2 import Environment, select_autoescape, PackageLoader


def get_dim(name):
    if "1d" in name:
        return 1
    elif "2d" in name:
        return 2
    elif "3d" in name:
        return 3
    else:
        return 0


def param_fix(f, param):
    conv_one = ["kernel_size", "stride", "dilation"]
    conv_zero = ["padding", "output_padding"]
    if get_dim(f) > 0:
        if param.name in conv_one:
            return (1,) * get_dim(f)
        elif param.name in conv_zero:
            return (0,) * get_dim(f)
    if ("in" in param.name or "out" in param.name) and param.default == inspect._empty:
        return 1
    return None


def get_namepar(obj):
    ans = {"name": obj.__name__, "params": dict(), "param_names": []}
    print(ans["name"], end="")
    sig = inspect.signature(obj.__init__)
    for k, v in sig.parameters.items():
        if k not in ["self", "args", "kwargs"]:
            ans["param_names"].append(k)
            ans["params"][v.name] = v.default if v.default != inspect._empty else None
            if isinstance(v.default, str):
                ans["params"][v.name] = '"' + v.default + '"'
            z = param_fix(ans["name"], v)
            if z is not None:
                print(" {}={} ".format(v.name, z), end="")
                ans["params"][v.name] = z
    print("")
    return ans


def extract_params(module_dict, folder_dest=None):
    if folder_dest is None:
        folder_dest = os.path.join(os.path.dirname(__file__), "../pytorch/")
    env = Environment(loader=PackageLoader("randonet", "templates"))
    temp = env.get_template("factory_template.py")
    fset = set()
    fdict = {}
    for k, v in module_dict.items():
        if k[0].isupper() and k != "F":
            if v.__module__ not in fset:
                fset.add(v.__module__)
                fdict[v.__module__] = []
            fdict[v.__module__].append(get_namepar(v))

    load_str = """
from randonet.generator.param import Param, IntParam, FloatParam, BinaryParam, ChoiceParam, TupleParam
from randonet.generator.unit import Unit, Factory as _Factory
from randonet.generator.conv import ConvFactory, ConvTransposeFactory
from collections import namedtuple

"""
    reqs = [
        "torch.nn.modules.linear",
        "torch.nn.modules.conv",
        "torch.nn.modules.activation",
        "torch.nn.modules.module",
        "torch.nn.modules.container",
    ]

    with open(os.path.join(folder_dest, "__init__.py"), "a+") as f:
        for name in fset - set(reqs):
            nm = name.split(".")[-1]
            with open(os.path.join(folder_dest, "{}.py".format(nm)), "w+") as f2:
                f2.write(load_str)
                fdict[name] = list(
                    sorted(fdict[name], key=lambda x: len(x["param_names"]))
                )
                for np in fdict[name]:
                    f2.write(temp.render(**np))
                    f2.write("\n\n")
            f.write(
                "from .{} import {}\n\n".format(
                    nm, ",".join(list(np["name"] for np in fdict[name]))
                )
            )


def extract_pytorch_params():
    extract_params(torch.nn.modules.__dict__, None)
