# Random Walks on Virtual Persistence Diagrams

This repository provides a Python implementation of translation-invariant heat flow random walks on virtual persistence diagram groups, as described in:

**Paper Title**: *Random Walks on Virtual Persistence Diagrams*

## Abstract

Persistence diagrams, with the bottleneck and Wasserstein metrics, represent the interval decomposition of persistence modules constructed from filtered topological data. Finite persistence diagrams on a metric pair $(X,A)$ form the free translation-invariant commutative Lipschitz monoid $D(X,A)$. The Grothendieck group $K(X,A)$ is the free translation-invariant Abelian Lipschitz group of virtual persistence diagrams, and the canonical embedding $D(X,A) \hookrightarrow K(X,A)$ is isometric for the Wasserstein--1 distance with translation-invariant metric $\rho$. When the pointed metric space $(X/A,\overline d_1,[A])$ is uniformly discrete, $(K(X,A),\rho)$ is a discrete locally compact abelian group, but may be uncountable. We construct a symmetric, translation-invariant Markov semigroup $(P_t)_{t\ge0}$ on $\ell^2(K(X,A))$ as the projective limit of semigroups defined on the finitely generated subgroups $K(X_F,A)$, induced by the VPD metric $\rho$. Its convolution kernels $(p_t)_{t\ge0}$ define a random walk on $K(X,A)$, and for each $t\ge0$ the support of $p_t$ is contained in a countable subgroup $H\le K(X,A)$. On $H$, the semigroup $(p_t)_{t\ge0}$ has a L\'evy--Khintchine representation $\widehat p_t(\theta)=\exp(-t\lambda_H(\theta))$ for $\theta\in\widehat H$, and the kernels $k_t(x,y)=p_t(x-y)$ define reproducing kernel Hilbert spaces $\mathcal H_t$ with dense truncated subspaces. We show that a small collection of scalar random--walk invariants determined by the L\'evy--Khintchine exponent---including return probabilities, collision probabilities, and diagonal resolvent values---controls global regularity properties of diagram functionals.

## Installation

### Requirements

- Python >= 3.8
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- NetworkX >= 3.0
- gudhi >= 3.7.0
- Matplotlib >= 3.7.0
- PyVista >= 0.40.0

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Basic Pipeline

```bash
python scripts/example/synthetic_pipeline.py
```

### Generate Figures

```bash
python scripts/example/generate_figures.py all
```

## Project Structure

```
.
├── src/
│   ├── vpd_group.py
│   ├── jump_kernel.py
│   ├── invariants.py
│   ├── bounds.py
│   ├── random_walk.py
│   └── pyvista_helpers.py
│
├── scripts/example/
└── results/example/
```

## Core Modules

### `src/vpd_group.py`
- Geometry window construction from BD points
- Mass functional $M(g) = \sum_u |n_u| \cdot d_1(u, [A])$
- 1-strengthened metric $d_1$ computation

### `src/jump_kernel.py`
- `LevyMeasure` class with jumps and rates
- Canonical geometry-induced Lévy measure construction
- Lévy-Khintchine exponent $\lambda_H(\theta)$

### `src/invariants.py`
- Return probability $p_t(0)$ via Monte Carlo integration
- Collision probability $\sum_h p_t(h)^2$
- Energy derivative $-\frac{d}{dt}p_t(0)$
- Resolvent diagonal $G_s(0,0)$

### `src/bounds.py`
- Lipschitz bounds: $Lip_\rho(f) \leq |f|_{H_t} \cdot (-\frac{d}{dt}p_t(0))^{1/2}$
- Ultracontractive bounds: $\|P_t\|_{2\to\infty} = \sqrt{\sum_h p_t(h)^2}$
- Sobolev-Green bounds using $G_s(0,0)$

### `src/random_walk.py`
- Compound Poisson simulation
- Translation-invariant heat flow random walks

## References

1. Edelsbrunner, H., Letscher, D., & Zomorodian, A. (2000). Topological persistence and simplification. *Proceedings 41st Annual Symposium on Foundations of Computer Science*, 454--463.
2. Zomorodian, A., & Carlsson, G. (2005). Computing persistent homology. *Discrete & Computational Geometry*, 33, 249--274.
3. Cohen-Steiner, D., Edelsbrunner, H., & Harer, J. (2007). Stability of persistence diagrams. *Discrete & Computational Geometry*, 37(1), 103--120.
4. Oudot, S. Y. (2015). *Persistence Theory: From Quiver Representations to Data Analysis*. Mathematical Surveys and Monographs, 209. American Mathematical Society.
5. Bubenik, P., & Elchesen, A. (2022). Virtual persistence diagrams, signed measures, Wasserstein distances, and Banach spaces. *Journal of Applied and Computational Topology*, 6, 429--474.
6. Fanning, C., & Aktas, M. (2025). Reproducing Kernel Hilbert Spaces for Virtual Persistence Diagrams. Preprint: https://arxiv.org/abs/2512.07282
7. Applebaum, D. (2009). *L\'evy Processes and Stochastic Calculus* (2nd ed.). Cambridge Studies in Advanced Mathematics, 116. Cambridge University Press.
8. Berg, C., & Forst, G. (1975). *Potential Theory on Locally Compact Abelian Groups*. Ergebnisse der Mathematik und ihrer Grenzgebiete, 87. Springer-Verlag.
9. Fanning, C., Aktas, M. (2026). *Reproducing Kernel Hilbert Spaces on Banach Comple-
tions of Virtual Persistence Diagram Groups* Preprint: https://arxiv.org/abs/2602.15153
10. Liggett, T. M. (2010). *Continuous Time Markov Processes: An Introduction*. Graduate Studies in Mathematics, 113. American Mathematical Society.
11. Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of "small-world" networks. *Nature*, 393(6684), 440--442.