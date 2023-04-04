//===- IndexingDialect.cpp - Indexing dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Gholi/IndexingDialect.h"
#include "Gholi/IndexingOps.h"
#include "Gholi/IndexingTypes.h"

using namespace mlir;
using namespace mlir::gholi;

#include "Gholi/IndexingOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Indexing dialect.
//===----------------------------------------------------------------------===//

void IndexingDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Gholi/IndexingOps.cpp.inc"
      >();
  registerTypes();
}
