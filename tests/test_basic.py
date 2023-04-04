from textwrap import dedent

from nelli.mlir.arith import constant
from nelli.mlir.func import mlir_func
from nelli.mlir.utils import I32, run_pipeline
from nelli.utils import mlir_mod_ctx

from gholi.mlir.passes import Pipeline
from util import check_correct
from gholi.mlir.dialects.indexing import FooOp, CustomType


class TestIfs:
    def test_smoke(self):
        with mlir_mod_ctx() as module:

            @mlir_func
            def foo(x: CustomType.get("bob")):
                cst = constant(2, type=I32)
                f = FooOp(cst)

        correct = dedent(
            """\
        module {
          func.func @foo(%arg0: !indexing.custom<"bob">) {
            %c2_i32 = arith.constant 2 : i32
            %0 = indexing.foo %c2_i32 : i32
            return
          }
        }
        """
        )
        check_correct(correct, module)

    def test_pass(self):
        with mlir_mod_ctx() as module:

            @mlir_func
            def bar():
                return

        run_pipeline(module, Pipeline().indexing_switch_bar_foo().materialize())

        assert "func.func @foo()" in str(module)
