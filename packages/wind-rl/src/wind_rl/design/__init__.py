"""Designer abstraction, feasible-layout geometry, and the layout buffer."""

from wind_rl.design.base import (
    Designer,
    DesignerConfig,
    FixedDesignerConfig,
    ManualDesignerConfig,
    RandomDesignerConfig,
)
from wind_rl.design.buffer import (
    LayoutConsumer,
    LayoutProducer,
    create_layout_buffer,
    run_buffer_dir,
)
from wind_rl.design.designers import (
    FixedDesigner,
    ManualDesigner,
    RandomDesigner,
    create_designer,
)
from wind_rl.design.geometry import (
    is_feasible,
    pairwise_min_distance,
    sample_feasible_layout,
    within_bounds,
)

__all__ = [
    "Designer",
    "DesignerConfig",
    "FixedDesigner",
    "FixedDesignerConfig",
    "LayoutConsumer",
    "LayoutProducer",
    "ManualDesigner",
    "ManualDesignerConfig",
    "RandomDesigner",
    "RandomDesignerConfig",
    "create_designer",
    "create_layout_buffer",
    "is_feasible",
    "pairwise_min_distance",
    "run_buffer_dir",
    "sample_feasible_layout",
    "within_bounds",
]
