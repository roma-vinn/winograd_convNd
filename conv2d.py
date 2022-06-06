import numpy as np
from math import ceil
from transform import Winograd2x3


def winograd_conv2d_layer_2x3(inp, weights):
    """Implementation of conv2d layer calculations using F(2, 3) Winograd algorithm for speed-up"""
    r = 3  # filter kernel size
    m = 2  # output size for WMFA
    alpha = m + r - 1  # tile size

    assert (
            weights.shape[2] == weights.shape[3] == r
    ), f"kernel must be have shape {r}x{r}, got {weights.shape[1:]}"

    # data transform matrix, 4x4
    B = Winograd2x3.B_m

    # filter transform matrix 4x3
    G = Winograd2x3.G_m

    # inverse transform matrix
    A = Winograd2x3.A_m

    N, C, H, W = inp.shape  # N - batch_size, C - channels
    K = weights.shape[0]  # number of kernels
    H_out, W_out = H - r + 1, W - r + 1  # output height and width

    # number of tiles along height and width
    tiles_h, tiles_w = ceil(H_out / m), ceil(W_out / m)
    P = N * tiles_h * tiles_w  # [per channel]

    # # prepare image tiles
    # d = np.empty(shape=(C, P, alpha, alpha))
    # for n in range(N):  # batch
    #     for c in range(C):  # channel
    #         for x in range(tiles_h):  # vertical index of tile
    #             for y in range(tiles_w):  # horizontal index of tile
    #                 p = n * (tiles_h * tiles_w) + x * tiles_w + y  # "number" of tile
    #                 # crop the tile of size (alpha, alpha) starting from (x, y) position
    #                 d[c, p] = inp[
    #                           n,
    #                           c,
    #                           x * (r - 1): x * (r - 1) + alpha,
    #                           y * (r - 1): y * (r - 1) + alpha,
    #                           ]

    # transform weights
    U = np.empty(shape=(K, C, alpha * alpha))
    for k in range(K):
        for c in range(C):
            u = np.dot(G, weights[k, c]).dot(G.T)
            # scatter intermediate results into matrices for xi/nu pairs
            U[k][c] = u.flatten()

    # transform inputs
    V = np.zeros(shape=(C, P, alpha * alpha))
    for n in range(N):
        for c in range(C):
            for x in range(tiles_h):  # vertical index of tile
                for y in range(tiles_w):  # horizontal index of tile
                    p = n * (tiles_h * tiles_w) + x * tiles_w + y  # "number" of tile
                    d = inp[
                              n,
                              c,
                              x * (r - 1): x * (r - 1) + alpha,
                              y * (r - 1): y * (r - 1) + alpha,
                              ]
                    v = np.dot(B.T, d).dot(B)

                    # scatter intermediate results into matrices for xi/nu pairs
                    V[c][p] = v.flatten()

    # perform point-wise multiplication for all tiles at ones
    # (by stacking them in matrices and performing dot product)
    M = np.empty(shape=(K, P, alpha * alpha))
    for i in range(alpha * alpha):
        M[:, :, i] = np.dot(U[:, :, i], V[:, :, i])

    # gather the result and perform inverse transformation
    tiles_out = np.empty(shape=(N, K, P, m, m))
    # tmp_m = np.empty(shape=(alpha, alpha))
    for n in range(N):  # over batches
        for k in range(K):  # over filters
            for p in range(P):  # over input tile
                # Gather m from matrices M
                tmp_m = M[k][p].reshape((alpha, alpha))
                tiles_out[n][k][p] = np.dot(A.T, tmp_m).dot(A)

    ans = np.empty(shape=(N, K, H_out, W_out))
    for n in range(N):
        for k in range(K):
            for x in range(tiles_h):
                for y in range(tiles_w):
                    p = n * (tiles_h * tiles_w) + x * tiles_w + y  # "number" of tile
                    ans[n, k, x * m: x * m + m, y * m: y * m + m] = tiles_out[n][k][p]
    return ans
