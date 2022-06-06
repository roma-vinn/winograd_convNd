# WMFA-based convolution for 4D and higher dimensions

This repository contains the code for the course work paper "WMFA-based convolution for 4D and higher dimensions".

Code structure:
- `baseline.py` – implementations of direct algorithm (considered as ground truth),
- `transform.py` contains transformation matrices for minimal F(2, 3) algorithm,
- `conv2d.py`, `conv4d.py`, `convNd.py` – implementations of ConvNet layers (forward pass) using Winograd convolution, 
- `test.py` – contains some simple tests.

After running `test.py` you should see output similar to this:

```
Testing ND[N=2] case:
Baseline approach time: 0.010751375000000007
Winograd approach time: 0.07086341699999998
Deviation from GT: np.mean(r_b_diff) = 6.925897016262033e-16
Testing 2D case:
Baseline approach time: 0.010915833999999958
Winograd approach time: 0.00936370799999997
Deviation from GT: np.mean(r_b_diff) = 6.925897016262033e-16
```