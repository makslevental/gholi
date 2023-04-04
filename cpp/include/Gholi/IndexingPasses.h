//===- IndexingPasses.h - Indexing passes  ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef INDEXING_INDEXINGPASSES_H
#define INDEXING_INDEXINGPASSES_H

#include "Gholi/IndexingDialect.h"
#include "Gholi/IndexingOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace gholi {
#define GEN_PASS_DECL
#include "Gholi/IndexingPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "Gholi/IndexingPasses.h.inc"



} // namespace indexing
} // namespace mlir

#endif
