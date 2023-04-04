from nelli.mlir.passes import Pipeline


class Pipeline(Pipeline):
    def indexing_switch_bar_foo(self):
        self._add_pass("indexing-switch-bar-foo")
        return self
