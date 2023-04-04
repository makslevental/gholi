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

# find the directory where the python bindings IRModule.h lives
# and add it as an include directory in order to extend/play with the bindings
get_target_property(mlir_python_sources MLIRPythonExtension.Core INTERFACE_SOURCES)
list(GET mlir_python_sources 0 HEADER)
get_filename_component(header_dir ${HEADER} DIRECTORY)
list(FILTER mlir_python_sources INCLUDE REGEX "(PybindUtils)\\.cpp")
target_sources(GholiPythonSources.Extension INTERFACE ${mlir_python_sources})
target_include_directories(GholiPythonSources.Extension INTERFACE ${header_dir})
