import numpy as np


def baseline_conv2d_layer(inp: np.ndarray, weights: np.ndarray) -> np.ndarray:
    B, C1, H, W = inp.shape
    K, C2, R1, R2 = weights.shape
    assert C1 == C2, "channels must be equal"
    assert R1 == R2 == 3, "kernels must be 3x3"
    R = R1

    conv = np.zeros((B, K, H - R + 1, W - R + 1))
    for b in range(B):
        for k in range(K):
            for y in range(H - R + 1):
                for x in range(H - R + 1):
                    conv[b, k, y, x] = np.sum(
                        inp[b, :, y : y + R, x : x + R] * weights[k, :, :, :]
                    )

    return conv


def baseline_conv2d(inp: np.ndarray, weight: np.ndarray) -> np.ndarray:
    a = np.reshape(inp, (1, 1, *inp.shape))
    b = np.reshape(weight, (1, 1, *weight.shape))
    return baseline_conv2d_layer(a, b)[0][0]


def baseline_conv3d_layer(inp: np.ndarray, weights: np.ndarray) -> np.ndarray:
    N, C1, H, W, D= inp.shape
    K, C2, R1, R2, R3 = weights.shape
    assert C1 == C2, "channels must be equal"
    assert R1 == R2 == R3 == 3, "kernels must be 3x3x3x3"
    R = R1

    conv = np.zeros((N, K, H - R + 1, W - R + 1, D - R + 1))
    for n in range(N):
        for k in range(K):
            for h in range(H - R + 1):
                for w in range(W - R + 1):
                    for d in range(D - R + 1):
                        conv[n, k, h, w, d] = np.sum(
                            inp[n, :, h : h + R, w : w + R, d : d + R]
                            * weights[k, :, :, :, :]
                        )

    return conv


def baseline_conv3d(inp: np.ndarray, weight: np.ndarray) -> np.ndarray:
    a = np.reshape(inp, (1, 1, *inp.shape))
    b = np.reshape(weight, (1, 1, *weight.shape))
    return baseline_conv3d_layer(a, b)[0][0]


def baseline_conv4d_layer(inp: np.ndarray, weights: np.ndarray) -> np.ndarray:
    N, C1, H, W, D, T = inp.shape
    K, C2, R1, R2, R3, R4 = weights.shape
    assert C1 == C2, "channels must be equal"
    assert R1 == R2 == R3 == R4 == 3, "kernels must be 3x3x3x3"
    R = R1

    conv = np.zeros((N, K, H - R + 1, W - R + 1, D - R + 1, T - R + 1))
    for n in range(N):
        for k in range(K):
            for h in range(H - R + 1):
                for w in range(W - R + 1):
                    for d in range(D - R + 1):
                        for t in range(T - R + 1):
                            conv[n, k, h, w, d, t] = np.sum(
                                inp[n, :, h : h + R, w : w + R, d : d + R, t : t + R]
                                * weights[k, :, :, :, :, :]
                            )

    return conv


def baseline_conv4d(inp: np.ndarray, weight: np.ndarray) -> np.ndarray:
    a = np.reshape(inp, (1, 1, *inp.shape))
    b = np.reshape(weight, (1, 1, *weight.shape))
    return baseline_conv4d_layer(a, b)[0][0]
