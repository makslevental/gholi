//===- IndexingTypes.h - Indexing dialect types -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef INDEXING_INDEXINGTYPES_H
#define INDEXING_INDEXINGTYPES_H

#include "mlir/IR/BuiltinTypes.h"

#define GET_TYPEDEF_CLASSES
#include "Gholi/IndexingOpsTypes.h.inc"

#endif // INDEXING_INDEXINGTYPES_H
