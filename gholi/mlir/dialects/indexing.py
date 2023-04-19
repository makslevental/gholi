#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Union, Optional

from nelli.mlir import DefaultContext
from nelli.mlir._mlir.ir import Value
import numpy as np
from numpy import ndindex
from .._mlir_libs._gholi_mlir import *

register_dialect(DefaultContext)

from ._indexing_ops_gen import *


def tensor_gather(source, indices, gather_dims):
    coordinate_size = indices.shape[-1]
    assert coordinate_size == len(gather_dims)
    out = np.empty(
        list(indices.shape[:-1])
        + [s for i, s in enumerate(source.shape) if i not in gather_dims],
    )

    def make_slice(x):
        slic = [slice(None)] * len(source.shape)
        for i, d in enumerate(gather_dims):
            slic[d] = x[i]
        return tuple(slic)

    for ii in ndindex(indices.shape[:-1]):
        idx = indices[ii]
        slic = make_slice(idx)
        val = source[slic]
        out[ii] = val

    return out


def tensor_scatter(source, destination, indices, scatter_dims):
    coordinate_size = indices.shape[-1]
    assert coordinate_size == len(scatter_dims)

    def make_slice(x):
        slic = [slice(None)] * len(source.shape)
        for i, d in enumerate(scatter_dims):
            slic[d] = x[i]
        return tuple(slic)

    for ii in ndindex(indices.shape[:-1]):
        idx = indices[ii]
        slic = make_slice(idx)
        val = source[slic]
        destination[ii] = val

    return destination
