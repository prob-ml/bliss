from typing import Dict, List, Sequence

from torchmetrics import Metric, MetricCollection


class SafeMetricCollection(MetricCollection):
    def __init__(
        self,
        metrics: Metric | Sequence[Metric] | Dict[str, Metric],
        *additional_metrics: Metric,
        prefix: str | None = None,
        postfix: str | None = None,
        compute_groups: bool | List[List[str]] = False,
    ) -> None:
        assert not compute_groups, "using compute_groups will lead to unexpected behavior"
        super().__init__(
            metrics,
            *additional_metrics,
            prefix=prefix,
            postfix=postfix,
            compute_groups=compute_groups,
        )
