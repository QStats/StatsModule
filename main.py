from QHyper.problems.community_detection import (
    CommunityDetectionProblem,
    KarateClubNetwork,
)
from QStats.louvain.louvain import Louvain
from Printer.printer import Printer
from util import G
import networkx as nx


name = "karate"
folder = "demo/network_community_detection/demo_output"
solution_file = f"{folder}/csv_files/{name}_adv_solution.csv"
decoded_solution_file = f"{folder}/csv_files/{name}_adv_decoded_solution.csv"
# img_solution_path = f"{folder}/{name}_louvain.png"


resolution = 0.5
problem = CommunityDetectionProblem(
    network_data=KarateClubNetwork(resolution=resolution), communities=2
)

res = Louvain.run_parallel(100, 0.5)

Printer.csv_from_array(res, solution_file)

samples = res["sample"]
print(samples)
for i, s in enumerate(samples):
    a = s[0]
    mod = res["mod_score"][i][0]
    path = f"{folder}/{name}_louvain_{i}.png"
    pos = nx.spring_layout(problem.G, seed=123);
    title = f"solver: {Louvain.solver} mod: {mod}"
    Printer.draw_nx(G, Printer.calculate_color_map_louvain(a), path, 
                    pos=pos,
                    title=title)
