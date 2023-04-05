from textwrap import dedent

import numpy as np
from nelli.mlir._mlir._mlir_libs._mlir.ir import Value
from nelli.mlir.arith import constant
from nelli.mlir.func import mlir_func
from nelli.mlir.tensor import tensor_dialect as tensor

from nelli.mlir.utils import I32, run_pipeline
from nelli.utils import mlir_mod_ctx

from gholi.mlir.dialects.indexing import GatherOp, CustomType, tensor_gather
from gholi.mlir.passes import Pipeline
from util import check_correct


class TestIfs:
    def test_smoke(self):
        with mlir_mod_ctx() as module:

            @mlir_func
            def foo(x: CustomType.get("bob")):
                cst = constant(2, type=I32)
                assert isinstance(cst, Value)
                source = tensor.EmptyOp([10, 10], I32)
                indices = tensor.EmptyOp([1, 2, 2], I32)
                f = GatherOp(source, indices, [0, 1])

        correct = dedent(
            """\
        module {
          func.func @foo(%arg0: !indexing.custom<"bob">) {
            %c2_i32 = arith.constant 2 : i32
            %0 = tensor.empty() : tensor<10x10xi32>
            %1 = tensor.empty() : tensor<1x2x2xi32>
            %2 = indexing.gather %0[%1] coordinates = [0, 1] : (tensor<10x10xi32>, tensor<1x2x2xi32>) -> tensor<1x2xi32>
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

    def test_gather(self):
        # %out = tensor.gather %source[%indices] gather_dims([0, 1, 2]) :
        #   (tensor<4x4x4xf32>, tensor<1x2x 3xindex>) -> tensor<1x2x 1x1x1xf32>

        source = np.random.randint(0, 100, (4, 4, 4))
        indices = np.random.randint(0, 4, (1, 2, 3))
        gather_dims = [0, 1, 2]

        res = tensor_gather(source, indices, gather_dims)
        assert res.shape == (1, 2)

        #  for 6 rows of 7 columns take from source along dim 1 everything else
        #  %out = tensor.gather %source[%indices] gather_dims([1]) :
        #    (tensor<3x4x5xf32>, tensor<6x7x 1xindex>) -> tensor<6x7x 3x1x5xf32>

        source = np.random.randint(0, 100, (3, 4, 5))
        indices = np.random.randint(0, 4, (6, 7, 1))
        gather_dims = [1]

        res = tensor_gather(source, indices, gather_dims)
        assert res.shape == (6, 7, 3, 5)

        #   %gathered = tensor.gather %dest[%indices_i32] gather_dims([1, 2]) unique:
        #     (tensor<4x5x6xf32>, tensor<1x3x2xi32>) -> tensor<1x3x4x1x1xf32>

        source = np.random.randint(0, 100, (4, 5, 6))
        indices = np.random.randint(0, 4, (1, 3, 2))
        gather_dims = [1, 2]

        res = tensor_gather(source, indices, gather_dims)
        assert res.shape == (1, 3, 4)

        # %0 = tensor.empty() : tensor<10x10xi32>
        # %1 = tensor.empty() : tensor<1x2x2xi32>
        # %2 = indexing.gather %0[%1] coordinates = [0, 1] : (tensor<10x10xi32>, tensor<1x2x2xi32>) -> tensor<1x2xi32>

        source = np.random.randint(0, 100, (10, 10))
        indices = np.random.randint(0, 4, (1, 2, 2))
        gather_dims = [0, 1]

        res = tensor_gather(source, indices, gather_dims)
        assert res.shape == (1, 2)
