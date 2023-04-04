//===- IndexingPasses.cpp - Indexing passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gholi/IndexingPasses.h"

namespace mlir {
namespace gholi {
#define GEN_PASS_DEF_INDEXINGSWITCHBARFOO
#include "Gholi/IndexingPasses.h.inc"

class IndexingSwitchBarFooRewriter : public OpRewritePattern<func::FuncOp> {
public:
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getSymName() == "bar") {
      rewriter.updateRootInPlace(op, [&op]() { op.setSymName("foo"); });
      return success();
    }
    return failure();
  }
};

class IndexingSwitchBarFoo
    : public impl::IndexingSwitchBarFooBase<IndexingSwitchBarFoo> {
public:
  using impl::IndexingSwitchBarFooBase<IndexingSwitchBarFoo>::IndexingSwitchBarFooBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<IndexingSwitchBarFooRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    auto result = applyPatternsAndFoldGreedily(getOperation(), patternSet);
    if (result.failed())
      assert(false && "IndexingSwitchBarFooRewriter failed.");
  }
};

} // namespace gholi
} // namespace mlir