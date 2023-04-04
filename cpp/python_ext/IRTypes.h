//===- IRTypes.h - Gholi ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GHOLI_IRTYPES_H
#define GHOLI_IRTYPES_H

#include <pybind11/pybind11.h>

#include "Gholi/IndexingTypes.h"

namespace py = pybind11;

namespace mlir::gholi {
void populateIRTypes(py::module &m);
}

#endif // GHOLI_IRTYPES_H
