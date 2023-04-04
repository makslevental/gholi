# Gholi

This project explores [structured codegen beyond rectangular arrays](https://discourse.llvm.org/t/rfc-structured-codegen-beyond-rectangular-arrays/64707/1).
Concretely, this means it implements an MLIR dialect called `Indexing`.
High-level the goal is to (natively) bring Triton-like programming to MLIR.