//===- IndexingDialect.td - Indexing dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef INDEXING_DIALECT
#define INDEXING_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Indexing dialect definition.
//===----------------------------------------------------------------------===//

def Indexing_Dialect : Dialect {
    let name = "indexing";
    let summary = "A indexing out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::gholi";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base indexing operation definition.
//===----------------------------------------------------------------------===//

class Indexing_Op<string mnemonic, list<Trait> traits = []> :
        Op<Indexing_Dialect, mnemonic, traits>;

#endif // INDEXING_DIALECT
