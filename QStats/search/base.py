from abc import ABC, abstractmethod
from typing import Any, TypedDict

import numpy as np

ParamGrid = TypedDict(
    "ParamGrid",
    {"resolution_grid": np.ndarray, "score_resolutions": np.ndarray},
)


class Search(ABC):
    def __init__(self, id: int, **kwargs: Any) -> None:
        pass

    def _check_param_grid_len(
        self, score_resolutions: np.ndarray, modularity_resolutions: np.ndarray
    ) -> Any:
        if len(score_resolutions) != len(modularity_resolutions):
            raise Exception(
                "Param grid objects must be of the same length,"
                + f"got of length {len(score_resolutions)}"
                + f"and {len(modularity_resolutions)} instead"
            )

    @abstractmethod
    def search_grid(
        self, param_grid: ParamGrid, n_runs_per_param: int, **kwargs: Any
    ) -> np.ndarray:
        ...
