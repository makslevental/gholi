add_mlir_dialect_library(MLIRIndexing
        IndexingTypes.cpp
        IndexingDialect.cpp
        IndexingOps.cpp
        IndexingPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Indexing

        DEPENDS
        MLIRIndexingOpsIncGen
        MLIRIndexingPassesIncGen)
