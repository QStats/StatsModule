import networkx as nx
from Printer.printer import Printer
from QStats.louvain.louvain import Louvain
from util import G

name = "karate"
folder = "demo/network_community_detection/demo_output/louvain"
solution_file = f"{folder}/csv_files/{name}_louvain_solution.csv"


resolution = 0.5

res = Louvain.run_parallel(100, 0.5)

Printer.csv_from_array(res, solution_file)

samples = res["sample"]
print(samples)
for i, s in enumerate(samples):
    a = s[0]
    mod = res["mod_score"][i][0]
    path = f"{folder}/{name}_louvain_{i}.png"
    pos = nx.spring_layout(G, seed=123)
    title = f"solver: {Louvain.solver} mod: {mod}"
    Printer.draw_nx(
        G, Printer.calculate_color_map_louvain(a), path, pos=pos, title=title
    )
