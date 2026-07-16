class _ContinuousDistribution:
    def sf(
        self,
        x: float,
        *shape_parameters: float,
        scale: float = ...,
    ) -> float: ...

norm: _ContinuousDistribution
chi2: _ContinuousDistribution
gamma: _ContinuousDistribution
