add_mlir_dialect(IndexingOps indexing)
add_mlir_doc(IndexingDialect IndexingDialect Indexing/ -gen-dialect-doc)
add_mlir_doc(IndexingOps IndexingOps Indexing/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS IndexingPasses.td)
mlir_tablegen(IndexingPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRIndexingPassesIncGen)