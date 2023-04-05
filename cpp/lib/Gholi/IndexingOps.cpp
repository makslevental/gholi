//===- IndexingOps.cpp - Indexing dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Gholi/IndexingOps.h"
#include "Gholi/IndexingDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"

#define GET_OP_CLASSES
#include "Gholi/IndexingOps.cpp.inc"

using namespace mlir;
using namespace mlir::tensor;

LogicalResult mlir::gholi::GatherOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {

  ArrayRef<int64_t> coordinates =
      attributes.get("coordinates").cast<mlir::DenseI64ArrayAttr>();
  RankedTensorType expectedResultType = mlir::tensor::GatherOp::inferResultType(
      // source
      operands[0].getType().cast<RankedTensorType>(),
      // indices
      operands[1].getType().cast<RankedTensorType>(), coordinates,
      /*rankReduced=*/true);
  inferredReturnTypes.assign({expectedResultType});
  return success();
}
