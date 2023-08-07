import numpy as np


name = "karate"
folder = "demo/network_community_detection/demo_output"
outfile = f"{folder}/{name}_array.npz"


try:
    npzfile = np.load(outfile, allow_pickle=True)
    # print(npzfile)
    # print(npzfile.files)
    r = npzfile["res"]
    # print(r)
except Exception as e:
    print(e)

# print("\n\n\n\n\n")
# for el in r:
#     print(r)

i, k, sample, mod, energy, t = r.flatten()[0]
print(i, type(i))
print(k, type(k))
print(sample, type(sample))
print(mod, type(mod))
print(energy, type(energy))
print(t, type(t))
n = r.dtype
print(n)

g = tuple(r)
print("\n\n", type(g))
print(g)
