import sys, os
from randonet.network import (
    LinearOnly,
    LinearAC,
    Conv1dOnly,
    Conv1dAC,
    Conv2dOnly,
    Conv2dAC,
    Conv3dOnly,
    Conv3dAC,
    Conv1dThenLinear,
    Conv2dThenLinear,
    Conv3dThenLinear,
    ResNetStyle,
)
from randonet.pytorch.activation import LogSoftmax
from jinja2 import Environment, select_autoescape, PackageLoader


ENV = Environment(loader=PackageLoader("randonet", "templates"))
TEMP = ENV.get_template("linear_template.py")

closer_gen = LogSoftmax()
closer_gen.dim.val = 1


gen_list = (
    LinearOnly(start_shape=(784,), stop_shape=(10,), depth=2),
    Conv1dOnly(depth=2),
    Conv2dOnly(depth=2),
    Conv3dOnly(depth=2),
)
bias_list = (
    LinearOnly(start_shape=(784,), stop_shape=(10,), depth=2, bias_prob=0.4),
    Conv1dOnly(depth=2, bias_prob=0.4),
    Conv2dOnly(depth=2, bias_prob=0.4),
    Conv3dOnly(depth=2, bias_prob=0.4),
)
ac_list = (
    LinearAC(start_shape=(784,), stop_shape=(10,), depth=2, bias_prob=0.3),
    Conv1dAC(depth=2, bias_prob=0.3),
    Conv2dAC(depth=2, bias_prob=0.3),
    Conv3dAC(depth=2, bias_prob=0.3),
)

comp_list = (
    Conv1dThenLinear(depth=2),
    Conv2dThenLinear(depth=2),
    Conv3dThenLinear(depth=2),
    ResNetStyle(start_shape=(1, 28, 28), stop_shape=(10,), depth=2),
)

start = 1
num_nets = 5


def write_template(info_dict, temp=TEMP):
    name = info_dict["name"]
    fname = name.lower()
    info_dict["layers"].append(closer_gen([10], [10]))
    with open(os.path.join("./samples/", fname + ".py"), "w+") as f:
        f.write(temp.render(**info_dict))
        f.write("\n")
    return "from .{} import {}\n".format(fname, name)


def gen(depth=2):
    l2 = []
    for g in gen_list + bias_list:
        g.depth = depth
        l = g(num_nets=num_nets, startnum=start)
        l2 = l2 + [write_template(x) for x in l]
    return l2


def ac(depth=2):
    l2 = []
    for a in LinearAC.ac.choices:
        for g in ac_list + comp_list:
            g.depth = depth
            l = g(num_nets=num_nets, startnum=start)
            l2 = l2 + [write_template(x) for x in l]
        Conv1dAC.ac.draw_next()
        LinearAC.ac.draw_next()
    return l2


def main():
    global start, num_nets
    l2 = []
    for depth in range(2, 7):
        l2 = l2 + ac(depth)
        l2 = l2 + gen(depth)
        start = start + num_nets

    print(len(l2), "networks generated")
    with open(os.path.join("./samples/", "__init__.py"), "w+") as f:
        for inc in l2:
            f.write(inc)
            # print(inc, end="")


if __name__ == "__main__":
    main()
