# -*- coding: utf-8 -*-
"""
    randonet.conv
    ~~~~~~~~~~~~~

    Common functionality for the Conv-type factory objects

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
from randonet.generator.unit import Unit, Factory as _Factory


class ConvBaseFactory(_Factory):
    def _fix_inshape(self, _in_shape):
        if len(_in_shape) != self.params.kernel_size.size + 1:
            # TODO: Fix factoring shape issue by padding? <25-11-19> #
            raise NotImplementedError("Need to fix shaping issues")
        return _in_shape


class ConvFactory(ConvBaseFactory):
    def _get_outshape(self, fn, in_shape):
        out_shape = [fn.out_channels]
        for s, k, strid, dil, pad in zip(
            in_shape[1:], fn.kernel_size, fn.stride, fn.dilation, fn.padding
        ):
            o = 1 + (s + 2 * pad - 1 - dil * (k - 1)) // strid
            out_shape.append(o)
        return out_shape

    def _lock_kernel(self, fn, in_shape, out_shape):
        ksize = []
        for s, o, strid, dil, pad in zip(
            in_shape[1:], out_shape[1:], fn.stride, fn.dilation, fn.padding
        ):
            k = 1 + (strid + s + 2 * pad - 1 - o * strid) // dil
            ksize.append(k)
        return fn.replace(
            in_channels=in_shape[0], out_channels=out_shape[0], kernel_size=tuple(ksize)
        )


class ConvTransposeFactory(ConvBaseFactory):
    def _get_outshape(self, fn, in_shape):
        out_shape = [fn.out_channels]
        for s, k, strid, dil, pad, opad in zip(
            in_shape[1:],
            fn.kernel_size,
            fn.stride,
            fn.dilation,
            fn.padding,
            fn.output_padding,
        ):
            o = (s - 1) * strid - 2 * pad + dil * (k - 1) + opad + 1
            out_shape.append(o)
        return out_shape

    def _lock_kernel(self, fn, in_shape, out_shape):
        ksize = []
        for s, o, strid, dil, pad, opad in zip(
            in_shape[1:],
            out_shape[1:],
            fn.stride,
            fn.dilation,
            fn.padding,
            fn.output_padding,
        ):
            k = 1 + (o - (s - 1) * strid + 2 * pad - opad - 1) // dil
            ksize.append(o)
        return fn.replace(
            in_channels=in_shape[0], out_channels=out_shape[0], kernel_size=tuple(ksize)
        )
