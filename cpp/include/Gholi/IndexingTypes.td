//===- IndexingTypes.td - Indexing dialect types -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef INDEXING_TYPES
#define INDEXING_TYPES

include "mlir/IR/AttrTypeBase.td"
include "Gholi/IndexingDialect.td"

//===----------------------------------------------------------------------===//
// Indexing type definitions
//===----------------------------------------------------------------------===//

class Indexing_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Indexing_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Indexing_CustomType : Indexing_Type<"Custom", "custom"> {
    let summary = "Indexing custom type";
    let description = "Custom type in indexing dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);
    let assemblyFormat = "`<` $value `>`";
}

#endif // INDEXING_TYPES
