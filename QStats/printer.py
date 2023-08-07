import csv
import os
from typing import Any

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from util import G

COLORS = {
    0: "blue",
    1: "red",
    2: "#2a401f",
    3: "#cce6ff",
    4: "pink",
    5: "#4ebd1a",
    6: "#66ff66",
    7: "yellow",
    8: "#0059b3",
    9: "#703243",
    10: "green",
    11: "black",
    12: "#3495eb",
    13: "#525c4d",
    14: "#1aff1a",
    15: "brown",
    16: "gray",
}


class Printer:
    @staticmethod
    def safe_open(path: str, permission: str) -> Any:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return open(path, permission)

    @staticmethod
    def csv_from_array(self, arr: np.ndarray, path: str) -> None:

        header = arr.dtype.names
        with self.safe_open(path, "w") as file:
            writer = csv.writer(file)
            writer.writerow(header)

            for i in range(arr.shape[0]):
                data = arr[i]
                writer.writerow(data)

    def _calculate_color_map(self, sample: dict, graph: nx.Graph):
        color_map = []
        if "x" in list(sample.keys())[0] or "s" in list(sample.keys())[0]:
            for node in graph:
                color_map.append(COLORS[sample["x" + str(node)]])
        else:
            for node in graph:
                color_map.append(COLORS[sample[str(node)]])

    def _draw_nx(
        self, graph: nx.Graph, color_map: list, path: str, **kwargs
    ) -> None:
        pos = kwargs.get("pos") if "pos" in kwargs else None
        f = plt.figure()
        nx.draw(
            graph,
            pos=pos,
            node_color=color_map,
            with_labels=True,
            ax=f.add_subplot(111),
        )
        if "title" in kwargs:
            plt.title(kwargs.get("title"))
        try:
            f.savefig(path)
        except Exception:
            plt.show()

    @staticmethod
    def draw_communities_from_sample(
        self, sample: dict, path: str, graph: nx.Graph = G
    ):
        color_map = self._calculate_color_map(sample, color_map, graph)
        self._draw_nx(graph, color_map, path)
