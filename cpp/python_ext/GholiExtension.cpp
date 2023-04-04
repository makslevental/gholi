#include "IRTypes.h"

#include "Gholi/IndexingDialect.h"
#include "Gholi/IndexingPasses.h"

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/FileSystem.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mlir;
using namespace mlir::gholi;

void registerGholi(MLIRContext &context) {
  DialectRegistry registry;
  registry.insert<mlir::gholi::IndexingDialect>();
  context.appendDialectRegistry(registry);
}

PYBIND11_MODULE(_gholi_mlir, m) {
  m.def(
      "register_dialect",
      [](const py::handle mlirContext) {
        auto *context = unwrap(mlirPythonCapsuleToContext(
            py::detail::mlirApiObjectToCapsule(mlirContext).ptr()));
        registerGholi(*context);
        context->getOrLoadDialect<mlir::gholi::IndexingDialect>();
      },
      py::arg("context") = py::none());

  registerIndexingSwitchBarFooPass();
  gholi::populateIRTypes(m);
}
