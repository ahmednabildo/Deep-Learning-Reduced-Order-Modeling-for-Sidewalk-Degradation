# Deep Learning–Based Reduced-Order Modeling for Sidewalk Elastodynamics

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmednabildo/Deep-Learning-Reduced-Order-Modeling-for-Sidewalk-Degradation/blob/main/Assignments_2.ipynb)

**POD-NN and DL-ROM surrogates for a time-dependent, non-affinely parameterized structural problem, with inverse parameter estimation from sparse surface measurements.**

This project compares two data-driven reduced-order modeling (ROM) approaches for predicting the displacement field of a two-layer sidewalk under a time-varying pedestrian load:

1. **POD-NN:** Proper Orthogonal Decomposition (POD) compresses the finite element solutions, and a neural network learns the map from physical parameters and time to POD coefficients [2, 3].
2. **DL-ROM:** a nonlinear autoencoder learns a low-dimensional representation, while a separate network predicts its latent coordinates from parameters and time [4].

The trained **POD-NN** is subsequently used as a fast surrogate for an **inverse problem**: estimating the material-interface location and loading magnitude from a small set of observed vertical surface deformations.

> **Scope:** The provided notebook studies a parameterized *linear-elastic elastodynamic* problem. "Sidewalk degradation" is the assignment's scenario; the notebook does not implement an explicit damage, cracking, or material-degradation evolution law.

## Results at a glance

The following figures are **saved outputs from [`Assignments_2.ipynb`](Assignments_2.ipynb)** [1]. They are not independently reproduced benchmarks, and no random seed is set in the notebook, so a fresh training run will give slightly different numbers.

| Quantity                                       | Notebook result                                          |
| ---------------------------------------------- | -------------------------------------------------------- |
| Full-order simulation trajectories             | 100                                                      |
| Time samples per trajectory                    | 51                                                       |
| Full-order displacement degrees of freedom     | 3,362                                                    |
| Training / test trajectories                   | 75 / 25                                                  |
| POD basis dimension                            | 30                                                       |
| Snapshot energy captured by 30 POD modes       | **99.99998%**                                            |
| POD reconstruction error on training snapshots | **0.09%**                                                |
| POD-NN test relative error (saved weights)     | **1.91%**                                                |
| DL-ROM test relative error                     | **3.91%**                                                |
| Total training time, POD-NN / DL-ROM           | ≈ 5.1 min / ≈ 9.1 min (GPU wall-clock, as printed)       |
| POD-NN inference for one 51-step trajectory    | **≈ 0.0043 s** (printed as 0.004 s)                      |
| Reported per-trajectory speedup                | **≈ 1,878×** (against an assumed 8 s full-order runtime) |

The relative errors use the notebook's $`L^2`$-based, trajectory-aggregated metric (Section 2). Results depend on the dataset, split, software environment, and trained weights.

## 1. Physical and numerical problem

The model represents a **two-layer, unit-square sidewalk**. A $`40 \times 40`$ finite element mesh and a first-order, vector-valued continuous Galerkin space are constructed using the `dlroms` finite element interface and FEniCS [7, 8]. The interface between the layers is at a variable vertical position $`\delta`$.

As stated in the notebook, the Lamé coefficients $`\lambda`$ and $`\eta`$ depend on the interface location through a step function in $`y`$ ($`\eta`$ is used so that it does not clash with the parameter vector $`\mu`$ below):

| Layer | Vertical region | Lamé coefficients (normalized) |
| ----- | --------------- | ------------------------------ |
| Upper | $`y \ge \delta`$  | $`\lambda = \eta = 10`$          |
| Lower | $`y < \delta`$    | $`\lambda = \eta = 50`$          |

The dataset uses two physical parameters, $`\mu = (\delta, m)`$, where $`\delta`$ is the **material-interface position** and $`m`$ controls the **magnitude of the pedestrian load** (the notebook's heatmap labels it "mass"). Time $`t`$ is a third input to the learned surrogate:

```math
(\delta, m, t) \;\longmapsto\; u(\delta, m, t) \quad \text{(full displacement field)}
```

As described in the notebook's discussion, the load is purely vertical, $`\mathbf{f}(t) = [0,\, -5m\sin(2\pi t)]^\top`$, active during a loading phase $`0 \le t \le 0.5`$ and zero afterwards (relaxation phase), with $`t`$ normalized to $`[0, 1]`$.

The notebook works with a supplied full-order dataset rather than generating all finite element trajectories itself. Its input tensors have shapes:

- **Parameters:** `(100, 2)`
- **Displacement trajectories:** `(100, 51, 3362)`
- **Flattened surrogate inputs:** `(5100, 3)` for $`(\delta, m, t)`$
- **Flattened full-order targets:** `(5100, 3362)`

For the inverse search, the notebook scans $`\delta \in [0.5, 0.9]`$ and $`m \in [0.5, 1.5]`$.

### Why use data-driven ROMs?

Projection-based POD–Galerkin relies on an affine decomposition of the operator,

```math
A(\mu) = \sum_q \theta_q(\mu)\, A_q,
```

with parameter-dependent scalars $`\theta_q(\mu)`$ and parameter-independent matrices $`A_q`$ assembled once offline. Here $`\delta`$ moves the interface *inside the integration domain*, so the matrices $`A_1(\delta)`$ (region $`y \ge \delta`$) and $`A_2(\delta)`$ (region $`y < \delta`$) change with every new $`\delta`$. The parametrization is **non-affine**, and POD–Galerkin would still have to reassemble the full stiffness matrix for every new $`(\delta, m)`$ unless additional treatment (for example hyper-reduction or empirical interpolation) is introduced.

This project instead learns the parameter-to-displacement relationship from snapshots, bypassing online finite element operator assembly [3, 4].

## 2. POD basis from simulation snapshots

The first **75 trajectories**, containing $`75 \times 51 = 3{,}825`$ snapshots, are used for model training; the remaining **25 trajectories** (1,275 snapshots) are reserved for testing. The split is by trajectory index; the notebook does not shuffle.

<!-- TODO: state whether the 100 parameter pairs in sidewalk.npz are randomly ordered. If they are sorted, a split by index could bias the test set (quick check: print(mu[:10])). -->

Let $`S_{\text{train}}`$ contain the training displacement snapshots as columns. Singular Value Decomposition (SVD) gives [2]:

```math
S_{\text{train}} = U\,\Sigma\,W^\top, \qquad V = U_{:,\,1:N}, \quad N = 30
```

The first $`N = 30`$ left singular vectors define the POD basis $`V`$. $`N`$ was chosen from the decay of the singular values and the cumulative captured energy

```math
E_N = \frac{\sum_{i=1}^{N}\sigma_i^2}{\sum_{i}\sigma_i^2}
```

The notebook reports **99.99998%** captured snapshot energy and **0.09%** POD reconstruction error on the training snapshots. The compressed representation of a displacement vector is

```math
c = V^\top u, \qquad u \approx V c
```

**Error metric.** The notebook's relative error is computed per trajectory as the ratio of time-summed $`L^2`$ norms (finite element $`L^2`$ norm, via `dlroms`) and then averaged over trajectories.

**Note on the two numbers.** For the plain Euclidean norm, the relative reconstruction error over the whole snapshot set equals $`\sqrt{1 - E_N}`$, which here is about 0.045%. The reported 0.09% is larger because it uses the $`L^2`$ (mass-matrix) norm and the per-trajectory aggregation above, whereas $`E_N`$ comes directly from the Euclidean SVD. The two figures are therefore not expected to match exactly.

## 3. POD-NN: neural prediction of POD coefficients

Instead of solving a reduced Galerkin system, a neural network predicts the 30 POD coordinates directly:

```math
\hat u(\delta, m, t) = V\,\hat c(\delta, m, t), \qquad \hat c = \mathcal{N}_\theta(\delta, m, t) \in \mathbb{R}^{30}
```

The implemented architecture consists of:

- **30 Fourier modes** applied to the time input using `dlroms.dnns.Fourier` (a Fourier-feature-style encoding [5]); the original three inputs are kept, giving a 63-dimensional input to the first dense layer;
- Dense layers of widths **256 → 512 → 256**, with GELU activations;
- A final linear layer producing **30 POD coefficients**;
- Adam optimization (batch size 128, He initialization, mean-squared loss on the POD coefficients) in three sequential learning-rate stages: **1e−3 (1,500 epochs)**, **1e−4 (500 epochs)**, and **1e−5 (500 epochs)**.

After the last stage the coefficient-space relative error (the training metric) is 1.55% on the training set and 1.84% on the test set. The final field-level evaluation reports a **1.91%** average relative error on the held-out trajectories for the saved weights. A run of the same evaluation directly after training printed 1.88%; the small difference reflects the unseeded training. The trained network and POD basis are saved in [`pod_nn.npz`](pod_nn.npz) and [`pod_basis.npy`](pod_basis.npy), respectively.

## 4. DL-ROM: nonlinear latent representation

The notebook also tests an autoencoder-based DL-ROM [4]. Instead of relying on a fixed POD basis, a neural encoder maps the 3,362-dimensional full-order displacement to a **five-dimensional latent space** ($`2p + 1`$ with $`p = 2`$ parameters):

```math
u \in \mathbb{R}^{3362} \;\xrightarrow{\text{encoder}}\; z \in \mathbb{R}^{5} \;\xrightarrow{\text{decoder}}\; \hat u \in \mathbb{R}^{3362}
```

- **Encoder:** $`3362 \to 256 \to 64 \to 5`$ (GELU).
- **Decoder:** $`5 \to 64 \to 256 \to 512 \to 1024 \to 3362`$ (GELU, linear output).

A separate network learns $`(\delta, m, t) \mapsto z`$ (time Fourier features with 30 modes, one hidden layer of 500 units, linear output). The resulting prediction is

```math
\hat z = \mathcal{P}_\phi(\delta, m, t), \qquad \hat u = \text{decoder}(\hat z)
```

Training uses Adam with batch size 128. The autoencoder is trained in two stages (1,500 epochs at 1e−3, then 500 epochs at 1e−4); the parameter-to-latent network is subsequently trained in two stages (1,500 epochs at 1e−3, then 50 epochs at 1e−4). The recorded full DL-ROM test error is **3.91%**.

The error budget is dominated by the parameter-to-latent map: the autoencoder alone reaches about 1.11% test error, whereas the parameter network reaches about 9.6% relative error in latent space, which gives the 3.91% end-to-end error.

**Comparison:** For this dataset and these saved experiments, **POD-NN performed better** than the tested DL-ROM (1.91% versus 3.91% test error) and trained faster (≈ 5.1 versus ≈ 9.1 minutes). The two models differ in more than the method: the DL-ROM uses 5 latent variables versus 30 POD modes, and the training budgets differ. This is a comparison of these implementations, not a general claim that POD-NN always outperforms nonlinear DL-ROMs.

## 5. Error analysis and inference performance

For the POD-NN, the notebook splits the test error into three parts, using sums of squared Euclidean errors over the 25 test trajectories (not normalized):

| Component                                                                  | Value  |
| -------------------------------------------------------------------------- | ------ |
| **Projection error:** $`\lVert u - VV^\top u \rVert^2`$ (information lost by the 30-mode POD space) | 0.0016 |
| **Coefficient-prediction error:** $`\lVert V^\top u - \hat c \rVert^2`$ (network error)           | 0.5012 |
| **Total field-prediction error:** $`\lVert u - V\hat c \rVert^2`$                                  | 0.5027 |

Because $`V`$ has orthonormal columns, the total is the sum of the other two, and more than 99% of it comes from the network's coefficient prediction, not from the POD truncation. Improving the POD-NN therefore means improving the regression, not adding modes.

For a sample **51-step trajectory**, the notebook prints a POD-NN evaluation time of 0.004 s (≈ 0.0043 s, consistent with the reported ratio). The **≈ 1,878× speedup** is calculated by comparing that runtime with a **hard-coded 8-second FOM reference** ($`8\,\text{s} / 0.0043\,\text{s} \approx 1{,}878`$), rather than timing the FOM in the same benchmark cell. The surrogate is timed with a single call on the GPU, without warm-up or explicit synchronization. It should therefore be treated as an illustrative speedup, not a controlled performance benchmark.

## 6. Inverse problem: recovering unknown physical parameters

The last part of the project uses the trained POD-NN model to estimate material and loading parameters from six observations of **average vertical deformation on the top boundary**, taken at the time indices $`0, 10, 20, 30, 40, 50`$ (normalized times $`0, 0.2, \dots, 1`$). The measurements (`surface_measurements.npy`) are supplied with the assignment; their noise level is not documented in the notebook.

Let $`Q(u)`$ extract those six observation values from a full trajectory, and let $`y_{\text{obs}}`$ denote the supplied measurements ($`\lVert y_{\text{obs}} \rVert_2 \approx 0.127`$). The inverse problem is formulated as a grid search:

```math
(\delta^*, m^*) = \arg\min_{(\delta,\, m)} \left\| Q\big(\hat u(\delta, m, \cdot)\big) - y_{\text{obs}} \right\|_2
```

Here, $`\hat u(\delta, m, \cdot)`$ denotes the **entire predicted time trajectory** for that parameter pair.

The notebook evaluates a **$`40 \times 40`$ parameter grid (1,600 pairs)** over $`\delta \in [0.5, 0.9]`$ and $`m \in [0.5, 1.5]`$ and plots the mismatch surface. The grid spacings are $`\Delta\delta = 0.4/39 \approx 0.010`$ and $`\Delta m = 1/39 \approx 0.026`$.

| Inverse-search result              | Notebook value                           |
| ---------------------------------- | ---------------------------------------- |
| Estimated interface position, $`\delta^*`$ | **≈ 0.777** (grid node 27 of 0–39) |
| Estimated load parameter, $`m^*`$    | **≈ 1.32** (grid node 32 of 0–39)        |
| Minimum mismatch norm              | **≈ 0.022**                              |
| Relative mismatch                  | **17.34%**                               |

Because the search is on a grid, $`\delta^*`$ and $`m^*`$ are only resolved to within the grid spacing above.

Two effects should be kept apart when reading these results:

1. **Weak identifiability.** The mismatch heatmap shows a diagonal valley in the $`(\delta, m)`$ plane. This is physically plausible: a larger load on a stiffer sidewalk (larger $`\delta`$ means more of the stiff lower layer) can give nearly the same average surface deformation as a smaller load on a softer one. The six observations alone therefore cannot uniquely identify both parameters.
2. **Non-zero residual at the minimum.** The relative mismatch at the best grid point is 17.34% (about 0.009 per observation, RMS), far above the 1.91% field-level test error. The notebook attributes this to the ROM error accumulating in $`Q`$, and states that refining the grid to $`80 \times 80`$ gave no change (that refinement is not part of the saved notebook). The notebook does not separate ROM error in $`Q`$ from measurement noise or model mismatch, so the cause of the residual is open.

<!-- TODO: once the two checks suggested for the notebook are run (ROM error on Q for test trajectories; synthetic inverse test), replace the last sentence of item 2 with the result. -->

The notebook discusses possible improvements: measuring the horizontal displacement $`u_x`$, concentrating measurements in the relaxation phase ($`t > 0.5`$), and using more observation times. These are **proposals**, not validated improvements in the current code. Inverse identification from limited, noisy observations is a general inverse-problem challenge [6].

## 7. Repository structure

```
.
├── Assignments_2.ipynb                 # Main notebook: POD, POD-NN, DL-ROM, inverse problem
├── Assignments 2 - Problems description.pdf
├── pod_basis.npy                       # Saved 30-mode POD basis
├── pod_nn.npz                          # Saved POD-NN model weights
└── README.md
```

The simulation dataset (`sidewalk.npz`) and the inverse measurements (`surface_measurements.npy`) are not stored in the repository; the notebook downloads them at run time (see Section 8).

The notebook visualizes the singular-value decay, training/test error histories, time-dependent vertical surface deformation, and an inverse-problem mismatch heatmap.

## 8. Software and how to run

**Main stack:** Python, PyTorch, NumPy, SciPy, Matplotlib, `dlroms`, FEniCS, `gdown`, and Jupyter/Google Colab [7, 8].

1. Set up a Python environment compatible with the `dlroms` and FEniCS versions used by your platform. The repository does **not** currently include a pinned environment or lockfile.
2. Open [`Assignments_2.ipynb`](Assignments_2.ipynb) in Jupyter or Google Colab (see the badge at the top).
3. Run the cells in order. The notebook uses `gdown` to retrieve the simulation dataset (`sidewalk.npz`) and inverse measurements (`surface_measurements.npy`) from external Drive file IDs included in the notebook.
4. The notebook explicitly calls `.cuda()` for the models and finite element norm, so **CUDA-enabled hardware/runtime is required as written**. CPU execution would require code changes.
5. Training regenerates the POD basis and model; alternatively, the repository includes the saved `pod_basis.npy` and `pod_nn.npz` artifacts, which the notebook's second part loads.

The initial notebook cell attempts to install `dlroms` from its upstream GitHub repository if import fails, using `pip install --no-deps`. That command **does not automatically install all dependent packages**, particularly compiled FEniCS dependencies.

## Acknowledgements

The problem statement, simulation dataset, and measurements were provided with the course assignment. The finite element and reduced-order-modeling tooling is the `dlroms` package by N. R. Franco [8].

## References

**Project sources**

[1] [`Assignments_2.ipynb`](Assignments_2.ipynb), project implementation and saved experimental outputs; and [`Assignments 2 - Problems description.pdf`](Assignments%202%20-%20Problems%20description.pdf), original problem statement.

**Scientific methods**

[2] Sirovich, L. (1987). "Turbulence and the dynamics of coherent structures. I. Coherent structures." *Quarterly of Applied Mathematics*, 45(3), 561–571. [DOI: 10.1090/S0033-569X-1987-0910462-6](https://doi.org/10.1090/S0033-569X-1987-0910462-6).

[3] Hesthaven, J. S., & Ubbiali, S. (2018). "Non-intrusive reduced order modeling of nonlinear problems using neural networks." *Journal of Computational Physics*, 363, 55–78. [DOI: 10.1016/j.jcp.2018.02.037](https://doi.org/10.1016/j.jcp.2018.02.037).

[4] Fresca, S., Dedè, L., & Manzoni, A. (2021). "A comprehensive deep learning-based approach to reduced order modeling of nonlinear time-dependent parametrized PDEs." *Journal of Scientific Computing*, 87, article 61. [DOI: 10.1007/s10915-021-01462-7](https://doi.org/10.1007/s10915-021-01462-7).

[5] Tancik, M., et al. (2020). "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains." *Advances in Neural Information Processing Systems (NeurIPS)*. [Official proceedings](https://proceedings.neurips.cc/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html).

[6] Vogel, C. R. (2002). *Computational Methods for Inverse Problems*. SIAM. [DOI: 10.1137/1.9780898717570](https://doi.org/10.1137/1.9780898717570).

**Software**

[7] Alnæs, M. S., et al. (2015). "The FEniCS Project Version 1.5." *Archive of Numerical Software*, 3(100). [DOI: 10.11588/ans.2015.100.20553](https://doi.org/10.11588/ans.2015.100.20553).

[8] Franco, N. R. (2024). *dlroms: A Python package for deep learning-based reduced-order models*. [Source code](https://github.com/NicolaRFranco/dlroms) · [Zenodo DOI: 10.5281/zenodo.13254758](https://doi.org/10.5281/zenodo.13254758).

---

*The quantitative results above reproduce those saved in the project notebook. Scientific citations provide background on the methods and software; they are not the source of the project-specific measurements.*
