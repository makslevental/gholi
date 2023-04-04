from nelli.mlir.passes.parse_mlir_opt_passes import (
    capture_help,
    fixup_lines_into_yaml,
    parse_passes,
)

before_gholi = capture_help()
# noinspection PyUnresolvedReferences
from gholi import mlir

after_gholi = capture_help()

yml = fixup_lines_into_yaml(list(set(after_gholi) - set(before_gholi)))
parse_passes(yml)
