include(AddMLIRPython)
mlir_configure_python_dev_packages()
mlir_detect_pybind11_install()

set(GHOLI_MLIR_PYTHON_PACKAGES_DIR "${CMAKE_CURRENT_BINARY_DIR}/python_packages")

# this can and should be set at runtime (in cpp using env vars or something?)
# see Interop.h
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=nelli.mlir._mlir.")

declare_mlir_python_sources(GholiPythonSources)
declare_mlir_dialect_python_bindings(
        ADD_TO_PARENT GholiPythonSources
        ROOT_DIR "${PROJECT_SOURCE_DIR}/gholi/mlir"
        TD_FILE dialects/IndexingOps.td
        SOURCES
        dialects/indexing.py
        DIALECT_NAME indexing)

declare_mlir_python_extension(GholiPythonSources.Extension
        MODULE_NAME _gholi_mlir
        ADD_TO_PARENT GholiPythonSources

        SOURCES
        GholiExtension.cpp
        IRTypes.cpp

        PRIVATE_LINK_LIBS
        MLIRIndexing)

set(_source_components GholiPythonSources)

add_mlir_python_modules(GholiPythonModules
        ROOT_PREFIX "${GHOLI_MLIR_PYTHON_PACKAGES_DIR}/gholi/mlir"
        INSTALL_PREFIX "."
        DECLARED_SOURCES ${_source_components})

set(PYBINDINGS_SRC "${MLIR_INSTALL_PREFIX}/src/python/MLIRPythonExtension.Core")
target_include_directories(GholiPythonSources.Extension INTERFACE "${PYBINDINGS_SRC}")
target_sources(GholiPythonSources.Extension INTERFACE "${PYBINDINGS_SRC}/PybindUtils.cpp")
