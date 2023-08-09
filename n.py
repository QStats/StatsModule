import numpy as np

N = 4

d_alias = ["ord", "k", "sample", "mod_score", "run_time"]
d_types = [np.int_, np.int_, np.object_, np.float64, np.float_]
types = np.dtype(
    [(a, d) for a, d in zip(d_alias, d_types)]
)
dims = (N, 1)
arr: np.ndarray = np.zeros(dims, dtype=types)

x = np.array([(1, 2), (3, 4)], dtype=[('foo', np.int_), ('bar', np.int_)])

p = np.array([[_] for _ in x['foo']]).squeeze()
print(p)
print(p.shape)


for i in range(N):
    arr[i] = i, i+1, {"d": 3, 4: "k"}, 0.6573853, 1232
print(f"ARR: {arr}\n\n\n")
print(arr.shape)

names = np.array(arr.dtype.names)
ids = arr["ord"]
k = arr["k"]
samples = arr["sample"]
modularity_scores = arr["mod_score"]
run_times = arr["run_time"]


# res = np.stack(
#     (tuple((ids, k, samples, modularity_scores, run_times))), axis=-1
# )

t = ids, k, samples, modularity_scores, run_times
s = np.array(t).squeeze()
# print(s.shape)
# print("\n\n\n\n\n\n\n", s)

stack = np.zeros(s.shape)

stack = np.hstack(
    (ids, k, samples, modularity_scores, run_times)
)
# print(names)
print(stack)
print(type(stack[0]))
print(stack[0].shape)
print(stack[0])
print(stack[0].reshape(1, -1).squeeze().shape)
for el in stack[0]:
    print(type(el))

print(stack.shape)
print(types)
print(f"stack shape: {stack.shape}")
res: np.ndarray = np.zeros(dims, dtype=types)
print(stack)
for i in range(res.shape[0]):
    r, s = res[i][0], stack[i]
    print(f"r: {r}, s: {s}")
    res[i][0] = tuple(s)


print(f"RES: {res}")
print(res.shape)
print(res.dtype.names)