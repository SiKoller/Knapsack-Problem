"""Validated settings and item data for the knapsack solver."""

from typing import Annotated, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt


class Item(NamedTuple):
    """Hold the weight and value of one item."""

    weight: StrictFloat
    value: StrictInt


class ProblemConfig(BaseModel):
    """Require the capacity and items from the problem file."""

    model_config = ConfigDict(frozen=True)

    capacity: Annotated[StrictFloat, Field(gt=0)]
    items: tuple[Item, ...]


class Config(BaseModel):
    """Check the solver settings."""

    model_config = ConfigDict(frozen=True)

    random_seed: int = 42
    penalty_for_overfill: float = -10.0
    epsilon: Annotated[float, Field(ge=0, le=1)] = 0.2
    alpha: Annotated[float, Field(gt=0, le=1)] = 0.1
    gamma: Annotated[float, Field(ge=0, le=1)] = 0.95
    episodes: Annotated[int, Field(gt=0)] = 500
    banner_width: Annotated[int, Field(gt=0)] = 50


CONFIG = Config()
