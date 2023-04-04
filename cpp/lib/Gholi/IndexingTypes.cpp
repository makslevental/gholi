//===- IndexingTypes.cpp - Indexing dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Gholi/IndexingTypes.h"

#include "Gholi/IndexingDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::gholi;

#define GET_TYPEDEF_CLASSES
#include "Gholi/IndexingOpsTypes.cpp.inc"

void IndexingDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Gholi/IndexingOpsTypes.cpp.inc"
      >();
}
