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
    def safe_open(path: str, permission: str, **kwargs) -> Any:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return open(path, permission, **kwargs)

    @staticmethod
    def csv_from_array(arr: np.ndarray, path: str) -> None:
        header = arr.dtype.names
        with Printer.safe_open(path, "w") as file:
            writer = csv.writer(file)
            writer.writerow(header)

            for i in range(arr.shape[0]):
                writer.writerow(*arr[i])

    @staticmethod
    def calculate_color_map(sample: dict, graph: nx.Graph) -> list:
        color_map = []
        if (
            "x" in str(list(sample.keys()))[0]
            or "s" in str(list(sample.keys()))[0]
        ):
            for node in graph:
                color_map.append(COLORS[int(sample["x" + str(node)])])
        else:
            for node in graph:
                color_map.append(COLORS[int(sample[node])])
        return color_map

    @staticmethod
    def calculate_color_map_louvain(communities: list, graph: nx.Graph = G):
        color_map = []
        for node in graph:
            if node in communities[0]:
                color_map.append("red")
            elif node in communities[1]:
                color_map.append("blue")
            else:
                color_map.append(COLORS[7])
        return color_map

    @staticmethod
    def draw_nx(graph: nx.Graph, color_map: list, path: str, **kwargs) -> None:
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
        sample: dict, path: str, graph: nx.Graph = G, **kwargs
    ):
        color_map = Printer.calculate_color_map(sample, graph)
        Printer.draw_nx(graph, color_map, path, **kwargs)

    @staticmethod
    def louvain_communities_to_sample_like(louvain_communities) -> dict:
        sample_like = {
            node_i: comm_i
            for comm_i, comms_set in enumerate(louvain_communities)
            for node_i in comms_set
        }
        return dict(sorted(sample_like.items()))

    @staticmethod
    def draw_samples_modularities(
        samples: np.ndarray,
        modularities: np.ndarray,
        base_path: str,
        solver: str,
    ):
        pos = (nx.spring_layout(G, seed=123),)
        for i, (s, m) in enumerate(zip(samples, modularities)):
            pos = nx.spring_layout(G, seed=123)
            title = f"solver: {solver} mod: {m[0]}"
            Printer.draw_communities_from_sample(
                s[0], f"{base_path}_{i}.png", pos=pos, title=title
            )
