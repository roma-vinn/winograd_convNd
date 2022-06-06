from time import perf_counter
import numpy as np


def test_conv(winograd_func, baseline_func, input_img, filters, **kwargs):
    t = perf_counter()
    baseline_res = baseline_func(input_img, filters)
    print(f'Baseline approach time: {perf_counter() - t}')

    t = perf_counter()
    res = winograd_func(input_img, filters, **kwargs)
    print(f'Winograd approach time: {perf_counter() - t}')

    r_b_diff = np.abs(res - baseline_res)
    print(f"Deviation from GT: {np.mean(r_b_diff) = }")


def test_2d():
    from baseline import baseline_conv2d_layer
    from convNd import winograd_convNd_layer_2x3
    from conv2d import winograd_conv2d_layer_2x3

    a = np.random.random((1, 3, 64, 64))
    b = np.random.random((1, 3, 3, 3))

    print('Testing 2D case:')
    test_conv(winograd_conv2d_layer_2x3, baseline_conv2d_layer, a, b)

    print('Testing ND[N=2] case:')
    test_conv(winograd_convNd_layer_2x3, baseline_conv2d_layer, a, b, Ndim=2)


def main():
    test_2d()


if __name__ == '__main__':
    main()
