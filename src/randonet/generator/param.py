# -*- coding: utf-8 -*-
"""
    randonet.is_randomr
    ~~~~~~~~~~~~~~~~~~~

    generate random things as per limits

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
import random


class Param(object):
    def __init__(self, name, default=None, is_random=False):
        self.name = name
        self.val = default
        self.is_random = is_random

    @property
    def value(self):
        if self.is_random:
            return self.generate()
        else:
            return self.default()

    def default(self):
        return self.val

    def generate(self):
        raise NotImplementedError("Base Class is not is_randomd")

    def _set_random(self, rval, **kwargs):
        self.is_random = rval
        for k, v in kwargs.items():
            if self.__dict__.get(k, None):
                setattr(self, k, v)

    def randomize(self, **kwargs):
        self._set_random(True, **kwargs)

    def unrandomize(self, **kwargs):
        self._set_random(False, **kwargs)

    def __call__(self, d):
        d[self.name] = self.value


class BinaryParam(Param):
    def __init__(self, name, default=True, is_random=False, true_prob=0.5):
        Param.__init__(self, name, default, is_random)
        self.true_prob = true_prob

    def generate(self):
        return random.uniform(0, 1) <= self.true_prob


class IntParam(Param):
    def __init__(self, name, default=1, is_random=False, limits=(1, 1)):
        Param.__init__(self, name, default, is_random)
        self.limits = limits

    def generate(self):
        if self.limits[0] == self.limits[1]:
            return self.limits[0]
        return random.randint(self.limits[0], self.limits[1])


class FloatParam(Param):
    def __init__(self, name, default=0.0, is_random=False, limits=(0.0, 0.0)):
        Param.__init__(self, name, default, is_random)
        self.limits = limits

    def generate(self):
        if self.limits[0] == self.limits[1]:
            return self.limits[0]
        return random.uniform(self.limits[0], self.limits[1])


class TupleParam(Param):
    def __init__(
        self, name, size, limits, is_random=False, default=None, is_square=True
    ):
        Param.__init__(self, name, (1,) * size, is_random)
        if default:
            self.val = default
        self.size = size
        self.limits = limits
        self.is_square = is_square

    def generate(self):
        ans = []
        if not self.is_square:
            for i in range(self.size):
                ans.append(random.randint(self.limits[0][i], self.limits[1][i]))
        else:
            ans = [random.randint(self.limits[0][0], self.limits[1][0])] * self.size
        return tuple(ans)


class ChoiceParam(Param):
    def __init__(self, name, choices, cprobs, is_random=False, default=None):
        Param.__init__(self, name, choices[0], is_random)
        if default:
            self.val = default
        self.choices = choices
        self.cprobs = cprobs

    def draw_next(self):
        self.val = self.choices[(self.choices.index(self.val) + 1) % len(self.choices)]

    def generate(self):
        x = random.uniform(0, 1)
        ans = 0
        for i in range(1, len(self.cprobs)):
            if x > self.cprobs[i - 1] and x <= self.cprobs[i]:
                ans = i
                break
        return self.choices[ans]
