import numpy as np
from math import ceil, prod
from transform import Winograd2x3
from tensorly.tenalg import multi_mode_dot
from time import perf_counter
from itertools import product, repeat


def winograd_convNd_layer_2x3(inp, weights, Ndim=4, verbose=0):
    """Implementation of convNd layer calculations using F(2, 3) Winograd algorithm for speed-up"""
    r = 3  # filter kernel size
    m = 2  # output size for WMFA
    alpha = m + r - 1  # tile size

    assert all(
        [weights.shape[i] == r for i in range(2, Ndim + 2)]
    ), f"kernel must be have shape {'x'.join(repeat(str(r), times=Ndim))}, got {weights.shape[2:]}"

    assert inp.shape[1] == weights.shape[1], f"input and weights channels must be equal"

    # data transform matrix, 4x4
    B = Winograd2x3.B_t

    # filter transform matrix 4x3
    G = Winograd2x3.G_t

    # inverse transform matrix
    A = Winograd2x3.A_t

    N, C, *W = inp.shape  # N - batch_size, C - channels
    K = weights.shape[0]  # number of kernels
    W_out = [w - r + 1 for w in W]  # output shapes

    # number of tiles along height and width
    tiles = [ceil(w_out / m) for w_out in W_out]
    P = N * prod(tiles)  # [per channel]
    # prod(tiles) = [tiles[0] * tiles[1] * ... * tiles[len(tiles) - 1]]

    weights_dot_time = 0
    t = perf_counter()
    # transform weights
    U = np.empty(shape=(K, C, alpha**Ndim))
    for k in range(K):
        for c in range(C):
            u = weights[k, c]

            c1 = perf_counter()
            u = multi_mode_dot(u, [G for _ in range(Ndim)])
            weights_dot_time += perf_counter() - c1

            U[k][c] = u.flatten()  # scatter intermediate results into vector
    if verbose:
        print(f"transform weights: {perf_counter() - t}")
        print(f"dot time = {weights_dot_time}")

    inputs_dot_time = 0
    t = perf_counter()
    # transform inputs
    V = np.zeros(shape=(C, P, alpha**Ndim))
    for n in range(N):
        for c in range(C):
            for indices in product(*map(range, tiles)):
                # "number" of tile
                p = n * prod(tiles) + sum([indices[i] * prod(tiles[i + 1 :]) for i in range(len(tiles))])

                slices = [slice(n, n + 1), slice(c, c + 1)] + [slice(i * (r - 1), i * (r - 1) + alpha) for i in indices]
                v = inp[tuple(slices)]
                v = v[0][0]

                c1 = perf_counter()
                v = multi_mode_dot(v, [B for _ in range(Ndim)])
                inputs_dot_time += perf_counter() - c1

                # scatter intermediate results into vector
                V[c][p] = v.flatten()
    if verbose:
        print(f"transform inputs: {perf_counter() - t}")
        print(f"dot time = {inputs_dot_time}")

    t = perf_counter()
    # perform point-wise multiplication for all tiles at ones
    # (by stacking them in matrices and performing dot product)
    M = np.empty(shape=(K, P, alpha**Ndim))
    for i in range(alpha**Ndim):
        M[:, :, i] = np.dot(U[:, :, i], V[:, :, i])
    if verbose:
        print(f"perform point-wise multiplication: {perf_counter() - t}")

    inverse_dot_time = 0
    t = perf_counter()
    # gather the result and perform inverse transformation
    tiles_out = np.empty(shape=(N, K, P, *[m for _ in range(Ndim)]))
    for n in range(N):  # over batches
        for k in range(K):  # over filters
            for p in range(P):  # over input tile
                # Gather m from matrix M
                tmp_m = M[k][p].reshape(tuple([alpha for _ in range(Ndim)]))

                c1 = perf_counter()
                out = tmp_m
                out = multi_mode_dot(out, [A for _ in range(Ndim)])
                inverse_dot_time += perf_counter() - c1

                tiles_out[n][k][p] = out
    if verbose:
        print(f"inverse transformation: {perf_counter() - t}")
        print(f"dot time = {inverse_dot_time}")

    t = perf_counter()
    ans = np.empty(shape=(N, K, *W_out))
    # combine those tiles back into convolution result
    for n in range(N):
        for k in range(K):
            for indices in product(*map(range, tiles)):
                # "number" of tile
                p = n * prod(tiles) + sum([indices[i] * prod(tiles[i + 1 :]) for i in range(len(tiles))])
                slices = [slice(n, n + 1), slice(k, k + 1)] + [slice(i * m, i * m + m) for i in indices]
                ans[tuple(slices)] = tiles_out[n][k][p]
    if verbose:
        print(f"combine tiles back together: {perf_counter() - t}")

    return ans
