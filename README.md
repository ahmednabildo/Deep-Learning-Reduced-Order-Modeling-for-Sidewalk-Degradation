# Deep Learning–Based Reduced-Order Modeling for Sidewalk Elastodynamics

**POD-NN and DL-ROM surrogates for a time-dependent, non-affinely parameterized structural problem, with inverse parameter estimation from sparse surface measurements.**

This project compares two data-driven reduced-order modeling (ROM) approaches for predicting the displacement field of a two-layer sidewalk under a time-varying pedestrian load:

1. **POD-NN:** Proper Orthogonal Decomposition (POD) compresses the finite element solutions, and a neural network learns the map from physical parameters and time to POD coefficients [2, 3].
2. **DL-ROM:** a nonlinear autoencoder learns a low-dimensional representation, while a separate network predicts its latent coordinates from parameters and time [4].

The trained **POD-NN** is subsequently used as a fast surrogate for an **inverse problem**: estimating the material-interface location and loading magnitude from a small set of observed vertical surface deformations.

> **Scope:** The provided notebook studies a parameterized *linear-elastic elastodynamic* problem. “Sidewalk degradation” is the assignment's scenario; the notebook does not implement an explicit damage, cracking, or material-degradation evolution law.

## Results at a glance

The following figures are **saved outputs from [`Assignments_2.ipynb`](Assignments_2.ipynb)** [1]. They are not independently reproduced benchmarks.

| Quantity | Notebook result |
| --- | ---: |
| Full-order simulation trajectories | 100 |
| Time samples per trajectory | 51 |
| Full-order displacement degrees of freedom | 3,362 |
| Training / test trajectories | 75 / 25 |
| POD basis dimension | 30 |
| Snapshot energy captured by 30 POD modes | **99.99998%** |
| POD reconstruction error on training snapshots | **0.09%** |
| POD-NN test relative error | **1.91%** |
| DL-ROM test relative error | **3.91%** |
| POD-NN inference for one 51-step trajectory | **~0.004 s** |
| Reported per-trajectory speedup | **1,878×** (against an assumed 8 s full-order runtime) |

The error percentages above use the notebook's L²-based, trajectory-aggregated evaluation where applicable. Results depend on the dataset, split, software environment, and trained weights.

## 1. Physical and numerical problem

The model represents a **two-layer, unit-square sidewalk**. A 40 × 40 finite element mesh and a first-order, vector-valued continuous Galerkin space are constructed using the `dlroms` finite element interface and FEniCS [7, 8]. The interface between the layers is at a variable vertical position, `δ`.

As stated in the notebook, the Lamé coefficients depend on the interface location:

| Layer | Vertical region | Lamé coefficients (normalized) |
| --- | --- | --- |
| Upper | `y ≥ δ` | `λ = η = 10` |
| Lower | `y < δ` | `λ = η = 50` |

The dataset uses two physical parameters, **`μ = (δ, m)`**, where **`δ` is the material-interface position** and **`m` controls the magnitude of the pedestrian load**. Time `t` is a third input to the learned surrogate:

```text
(delta, m, t)  -->  u(delta, m, t)  [full displacement field]
```

The notebook works with a supplied full-order dataset rather than generating all finite element trajectories itself. Its input tensors have shapes:

- **Parameters:** `(100, 2)`
- **Displacement trajectories:** `(100, 51, 3362)`
- **Flattened surrogate inputs:** `(5100, 3)` for `(δ, m, t)`
- **Flattened full-order targets:** `(5100, 3362)`

For the inverse search, the notebook scans `δ ∈ [0.5, 0.9]` and `m ∈ [0.5, 1.5]`. Time is normalized to `[0, 1]` for the neural network inputs.

### Why use data-driven ROMs?

The material interface moves with `δ`, so the parameter changes the *spatial distribution* of the elastic coefficients. A simple, fixed affine separation of the stiffness operator is therefore not directly available. Standard projection-based POD–Galerkin evaluation would need additional treatment of the non-affine operator (for example, hyper-reduction or empirical interpolation) to avoid high-dimensional online assembly.

This project instead learns the parameter-to-displacement relationship from snapshots, bypassing online finite element operator assembly [3, 4].

## 2. POD basis from simulation snapshots

The first **75 trajectories**, containing `75 × 51 = 3,825` snapshots, are used for model training; the remaining **25 trajectories** (1,275 snapshots) are reserved for testing.

Let `S_train` contain the training displacement snapshots as columns. Singular Value Decomposition (SVD) gives [2]:

```text
S_train = U · Σ · Wᵀ
```

The first `N = 30` left singular vectors define the POD basis `V`. Its captured-energy ratio is

```text
E_N = (σ₁² + σ₂² + ... + σ_N²) / (σ₁² + σ₂² + ...)
```

The notebook reports **99.99998%** captured snapshot energy and **0.09%** POD reconstruction error on the training snapshots. The compressed representation of a displacement vector is

```text
c = Vᵀ u             # Project a displacement onto the POD basis
u_approx = V c       # Reconstruct the full displacement
```

## 3. POD-NN: neural prediction of POD coefficients

Instead of solving a reduced Galerkin system, a neural network predicts the 30 POD coordinates directly:

```text
Input: (delta, m, t)
        |
        v
   Neural network
        |
        v
   c_hat (30 POD coefficients)
        |
        v
   u_hat = V @ c_hat
```

The implemented architecture consists of:

- **30 Fourier modes** applied to the time input using `dlroms.dnns.Fourier` (a Fourier-feature-style encoding [5]);
- Dense layers of widths **256 → 512 → 256**, with GELU activations;
- A final linear layer producing **30 POD coefficients**;
- Adam optimization with three sequential learning-rate stages: **1e−3 (1,500 epochs)**, **1e−4 (500 epochs)**, and **1e−5 (500 epochs)**.

The final notebook evaluation reports a **1.91%** average relative error on the held-out trajectories. The trained network and POD basis are saved in [`pod_nn.npz`](pod_nn.npz) and [`pod_basis.npy`](pod_basis.npy), respectively.

## 4. DL-ROM: nonlinear latent representation

The notebook also tests an autoencoder-based DL-ROM [4]. Instead of relying on a fixed POD basis, a neural encoder maps the 3,362-dimensional full-order displacement to a **five-dimensional latent space**:

```text
u (3,362 DOFs)
     |
     v
   Encoder
     |
     v
z (5 latent variables)
     |
     v
   Decoder
     |
     v
u_hat (3,362 DOFs)
```

A separate network learns `(δ, m, t) → z`, using time Fourier features. The resulting prediction is

```text
z_hat = parameter_net(delta, m, t)
u_hat = decoder(z_hat)
```

The autoencoder is trained in two stages (1,500 then 500 epochs); the parameter-to-latent network is subsequently trained in two stages (1,500 then 50 epochs). The recorded full DL-ROM test error is **3.91%**.

**Comparison:** For this dataset and these saved experiments, **POD-NN performed better** than the tested DL-ROM (1.91% versus 3.91% test error). This is a comparison of these implementations, not a general claim that POD-NN always outperforms nonlinear DL-ROMs.

## 5. Error analysis and inference performance

The notebook evaluates the models with an L²-based trajectory error. It also examines the POD-NN reconstruction-error decomposition into:

- **Projection error:** loss of information when projecting a high-dimensional solution onto the 30-mode POD space;
- **Coefficient-prediction error:** difference between exact projected coefficients and those predicted by the neural network;
- **Total field-prediction error:** difference between the true full-order solution and the reconstructed neural surrogate output.

For a sample **51-step trajectory**, the notebook records POD-NN evaluation time of approximately **0.004 seconds**. Its **1,878× speedup** is calculated by comparing that runtime with a **hard-coded 8-second FOM reference** in the notebook, rather than timing the FOM in the same benchmark cell. It should therefore be treated as an illustrative speedup, not a controlled performance benchmark.

## 6. Inverse problem: recovering unknown physical parameters

The last part of the project uses the trained POD-NN model to estimate material and loading parameters from six observations of **average vertical deformation on the top boundary**.

Let `Q(u)` extract those six observation values from a full trajectory, and let `y_obs` denote the supplied measurements. The inverse problem is formulated as a grid search:

```text
(delta*, m*) = argmin over (delta, m) of:
               || Q(u_hat(delta, m)) - y_observed ||₂
```

Here, `u_hat(delta, m)` denotes the **entire predicted time trajectory** for that parameter pair.

The notebook evaluates a **40 × 40 parameter grid (1,600 pairs)** over `δ ∈ [0.5, 0.9]` and `m ∈ [0.5, 1.5]` and plots the mismatch surface.

| Inverse-search result | Notebook value |
| --- | ---: |
| Estimated interface position, `δ*` | **0.776923** |
| Estimated load parameter, `m*` | **1.320513** |
| Minimum mismatch norm | **0.021982** |
| Relative mismatch | **17.34%** |

The mismatch heatmap contains an elongated valley, suggesting that different interface–load combinations can yield similar observations. This is an **identifiability limitation** of sparse observations, not proof of a unique physical parameter estimate. The notebook discusses possible improvements, including additional horizontal-displacement and temporal measurements. These are **proposals**, not validated improvements in the current code. Inverse identification from limited, noisy observations is a general inverse-problem challenge [6].

## 7. Repository structure

```text
.
├── Assignments_2.ipynb                 # Main notebook: POD, POD-NN, DL-ROM, inverse problem
├── Assignments 2 - Problems description.pdf
├── pod_basis.npy                       # Saved 30-mode POD basis
├── pod_nn.npz                          # Saved POD-NN model weights
└── README.md
```

The notebook visualizes the singular-value decay, training/test error histories, time-dependent vertical surface deformation, and an inverse-problem mismatch heatmap.

## 8. Software and how to run

**Main stack:** Python, PyTorch, NumPy, SciPy, Matplotlib, `dlroms`, FEniCS, `gdown`, and Jupyter/Google Colab [7, 8].

1. Set up a Python environment compatible with the `dlroms` and FEniCS versions used by your platform. The repository does **not** currently include a pinned environment or lockfile.
2. Open [`Assignments_2.ipynb`](Assignments_2.ipynb) in Jupyter or Google Colab.
3. Run the cells in order. The notebook uses `gdown` to retrieve the simulation dataset (`sidewalk.npz`) and inverse measurements (`surface_measurements.npy`) from external Drive file IDs included in the notebook.
4. The notebook explicitly calls `.cuda()` for the models and finite element norm, so **CUDA-enabled hardware/runtime is required as written**. CPU execution would require code changes.
5. Training regenerates the POD basis and model; alternatively, the repository includes the saved `pod_basis.npy` and `pod_nn.npz` artifacts used later in the notebook.

The initial notebook cell attempts to install `dlroms` from its upstream GitHub repository if import fails, using `pip install --no-deps`. That command **does not automatically install all dependent packages**, particularly compiled FEniCS dependencies.


## References

**Project sources**

[1] [`Assignments_2.ipynb`](Assignments_2.ipynb), project implementation and saved experimental outputs; and [`Assignments 2 - Problems description.pdf`](Assignments%202%20-%20Problems%20description.pdf), original problem statement.

**Scientific methods**

[2] Sirovich, L. (1987). “Turbulence and the dynamics of coherent structures. I. Coherent structures.” *Quarterly of Applied Mathematics*, 45(3), 561–571. [DOI: 10.1090/S0033-569X-1987-0910462-6](https://doi.org/10.1090/S0033-569X-1987-0910462-6).

[3] Hesthaven, J. S., & Ubbiali, S. (2018). “Non-intrusive reduced order modeling of nonlinear problems using neural networks.” *Journal of Computational Physics*, 363, 55–78. [DOI: 10.1016/j.jcp.2018.02.037](https://doi.org/10.1016/j.jcp.2018.02.037).

[4] Fresca, S., Dedè, L., & Manzoni, A. (2021). “A comprehensive deep learning-based approach to reduced order modeling of nonlinear time-dependent parametrized PDEs.” *Journal of Scientific Computing*. [DOI: 10.1007/s10915-021-01462-7](https://doi.org/10.1007/s10915-021-01462-7).

[5] Tancik, M., et al. (2020). “Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains.” *Advances in Neural Information Processing Systems (NeurIPS)*. [Official proceedings](https://proceedings.neurips.cc/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html).

[6] Vogel, C. R. (2002). *Computational Methods for Inverse Problems*. SIAM. [DOI: 10.1137/1.9780898717570](https://doi.org/10.1137/1.9780898717570).

**Software**

[7] Alnæs, M. S., et al. (2015). “The FEniCS Project Version 1.5.” *Archive of Numerical Software*, 3(100). [DOI: 10.11588/ans.2015.100.20553](https://doi.org/10.11588/ans.2015.100.20553).

[8] Franco, N. R. (2024). *dlroms: A Python package for deep learning-based reduced-order models*. [Source code](https://github.com/NicolaRFranco/dlroms) · [Zenodo DOI: 10.5281/zenodo.13254758](https://doi.org/10.5281/zenodo.13254758).

---

*The quantitative results above reproduce those saved in the project notebook. Scientific citations provide background on the methods and software; they are not the source of the project-specific measurements.*
