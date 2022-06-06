import numpy as np
from math import ceil
from tensorly.tenalg import multi_mode_dot
from transform import Winograd2x3
from time import perf_counter


def winograd_conv4d_layer_2x3(inp, weights, verbose=0):
    """Implementation of conv4d layer calculations using F(2, 3) Winograd algorithm for speed-up"""
    r = 3  # filter kernel size
    m = 2  # output size for WMFA
    alpha = m + r - 1  # tile size
    Ndim = 4

    assert (
        weights.shape[2]
        == weights.shape[3]
        == weights.shape[4]
        == weights.shape[5]
        == r
    ), f"kernel must be have shape {r}x{r}x{r}x{r}, got {weights.shape[2:]}"

    assert (
        inp.shape[1] == weights.shape[1]
    ), f"channels in input and weights must be equal"

    # data transform matrix, 4x4
    B = Winograd2x3.B_t

    # filter transform matrix 4x3
    G = Winograd2x3.G_t

    # inverse transform matrix
    A = Winograd2x3.A_t

    N, C, H, W, D, T = inp.shape  # N - batch_size, C - channels
    K = weights.shape[0]  # number of kernels
    H_out, W_out, D_out, T_out = (
        H - r + 1,
        W - r + 1,
        D - r + 1,
        T - r + 1,
    )  # output height and width

    # number of tiles along height and width
    tiles_h, tiles_w, tiles_d, tiles_t = (
        ceil(H_out / m),
        ceil(W_out / m),
        ceil(D_out / m),
        ceil(T_out / m),
    )
    P = N * tiles_h * tiles_w * tiles_d * tiles_t  # [per channel]

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
            # scatter intermediate results into matrices for xi/nu/ro triplets
            U[k][c] = u.flatten()
    if verbose:
        print(f"transform weights: {perf_counter() - t}")
        print(f"dot time = {weights_dot_time}")

    inputs_dot_time = 0
    t = perf_counter()
    # transform inputs
    V = np.zeros(shape=(C, P, alpha**Ndim))
    for n in range(N):
        for c in range(C):
            for i_h in range(tiles_h):  # vertical index of a tile
                for i_w in range(tiles_w):  # horizontal index of a tile
                    for i_d in range(tiles_d):  # depth index of a tile
                        for i_t in range(tiles_t):  # time index of a tile
                            # "number" of tile
                            b = (
                                n * (tiles_h * tiles_w * tiles_d * tiles_t)
                                + i_h * tiles_w * tiles_d * tiles_t
                                + i_w * tiles_d * tiles_t
                                + i_d * tiles_t
                                + i_t
                            )

                            v = inp[
                                n,
                                c,
                                i_h * (r - 1) : i_h * (r - 1) + alpha,
                                i_w * (r - 1) : i_w * (r - 1) + alpha,
                                i_d * (r - 1) : i_d * (r - 1) + alpha,
                                i_t * (r - 1) : i_t * (r - 1) + alpha,
                            ]

                            c1 = perf_counter()
                            v = multi_mode_dot(v, [B for _ in range(Ndim)])
                            inputs_dot_time += perf_counter() - c1

                            # scatter intermediate results into matrices for xi/nu/ro triplets
                            V[c][b] = v.flatten()

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

    dot_time = 0
    t = perf_counter()
    # gather the result and perform inverse transformation
    tiles_out = np.empty(shape=(N, K, P, m, m, m, m))
    for n in range(N):  # over batches
        for k in range(K):  # over filters
            for b in range(P):  # over input tile
                # Gather m from matrices M
                tmp_m = M[k][b].reshape((alpha, alpha, alpha, alpha))

                c1 = perf_counter()

                out = tmp_m
                out = multi_mode_dot(out, [A for _ in range(Ndim)])

                dot_time += perf_counter() - c1

                tiles_out[n][k][b] = out

    if verbose:
        print(f"inverse transformation: {perf_counter() - t}")
        print(f"dot time = {dot_time}")

    t = perf_counter()
    ans = np.empty(shape=(N, K, H_out, W_out, D_out, T_out))
    for n in range(N):
        for k in range(K):
            for i_h in range(tiles_h):  # vertical index of a tile
                for i_w in range(tiles_w):  # horizontal index of a tile
                    for i_d in range(tiles_d):  # depth index of a tile
                        for i_t in range(tiles_t):  # time index of a tile
                            # "number" of tile
                            b = (
                                n * (tiles_h * tiles_w * tiles_d * tiles_t)
                                + i_h * tiles_w * tiles_d * tiles_t
                                + i_w * tiles_d * tiles_t
                                + i_d * tiles_t
                                + i_t
                            )
                            ans[
                                n,
                                k,
                                i_h * m : i_h * m + m,
                                i_w * m : i_w * m + m,
                                i_d * m : i_d * m + m,
                                i_t * m : i_t * m + m,
                            ] = tiles_out[n][k][b]
    if verbose:
        print(f"combine tiles back together: {perf_counter() - t}")

    return ans
