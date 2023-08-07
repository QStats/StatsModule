import numpy as np

d_alias = ["ord", "k", "sample", "mod_score", "energy", "run_time"]
d_types = [np.int_, np.int_, np.object_, np.float64, np.float_]
types = np.dtype([(a, d) for a, d in zip(d_alias, d_types)])
dims = (1, 2)
arr: np.ndarray = np.zeros(dims, dtype=types)
arr[0, 0]["sample"] = {"ala": "ma", "kota": 2}
# print(arr)
arr[0, 1]["sample"] = {1: 3, "ala": object}
print(type(arr[0, 1]["sample"]))
# print(arr)
# print(f"arr:\n\n {arr}\n\n\n")
# arr[0][1] = "hello", [80, 88, 98]
# arr[0, 1]['s'] = "hi"
# arr[0,1]['fs'][1] = "2223"
# # arr[0] = 78.0
# # arr[1,0] = 99
# print(arr.flatten())
print(arr["sample"].flatten())
print(arr["sample"].flatten()[1])

print()
print(arr)

# outfile = TemporaryFile()
name = "karate"
folder = "demo/network_community_detection/demo_output"
outfile = f"{folder}/{name}_array.npz"
np.savez(outfile, x=arr)

try:
    npzfile = np.load(outfile)
except Exception:
    npzfile = np.load(outfile, allow_pickle=True)
finally:
    print(arr, "\n\n\n\n")

try:
    a = npzfile.files
    print(a)
finally:
    print(arr)
