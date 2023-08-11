import numpy as np

# da = ["matrix_res", "score_res"]
# dt = [np.float_, np.float_]
# n_params = 7
# types = np.dtype(
#     [(a, t) for a, t in zip(da, dt)]
# )
# res_val = 2
# mod_res = 4

# x = np.ndarray((n_params, ), dtype=types)
# x[::] = (res_val, mod_res)

# print(x)
# print(x.shape)
# print(x.dtype.names)
# print(x[2])

modularity_cols[::] = (res_val, mod_res)
print(f"modularity_cols: {modularity_cols}")

res_stacked = np.hstack([modularity_cols, *runs_res])
res = [
    [
        (
            2.0,
            {
                0: 1,
                1: 1,
                2: 0,
                3: 0,
                4: 1,
                5: 1,
                6: 1,
                7: 1,
                8: 1,
                9: 1,
                10: 1,
                11: 1,
                12: 1,
                13: 1,
                14: 1,
                15: 1,
                16: 1,
                17: 1,
                18: 1,
                19: 1,
                20: 1,
                21: 1,
                22: 0,
                23: 0,
                24: 0,
                25: 0,
                26: 0,
                27: 0,
                28: 1,
                29: 1,
                30: 0,
                31: 0,
                32: 0,
                33: 1,
                34: 0,
                35: 0,
                36: 1,
                37: 1,
                38: 1,
                39: 1,
                40: 1,
                41: 1,
                42: 0,
                43: 0,
                44: 0,
                45: 0,
                46: 0,
                47: 0,
                48: 0,
                49: 0,
                50: 0,
                51: 0,
                52: 0,
                53: 0,
                54: 1,
                55: 0,
                56: 1,
                57: 1,
                58: 0,
                59: 0,
                60: 1,
                61: 1,
                62: 1,
                63: 1,
                64: 0,
                65: 0,
                66: 0,
                67: 0,
                68: 1,
                69: 1,
                70: 1,
                71: 1,
                72: 1,
                73: 1,
                74: 1,
                75: 1,
                76: 1,
                77: 1,
                78: 1,
                79: 1,
                80: 1,
                81: 1,
                82: 1,
                83: 1,
                84: 0,
                85: 0,
                86: 1,
                87: 1,
                88: 0,
                89: 0,
            },
            0.46081067,
            -110.0296837,
            62.957932,
        )
    ]
]
modularity_cols = np.ndarray((n_runs_per_param,), dtype=types)
modularity_cols = [(0.8, 0.8)]
# modularity_cols =
res = np.array(res)
modularity_cols = np.array(modularity_cols)
print(res)
print(modularity_cols)
print(res.shape)
print(modularity_cols.shape)
stack = np.hstack([modularity_cols, *res])
print(stack.shape)
