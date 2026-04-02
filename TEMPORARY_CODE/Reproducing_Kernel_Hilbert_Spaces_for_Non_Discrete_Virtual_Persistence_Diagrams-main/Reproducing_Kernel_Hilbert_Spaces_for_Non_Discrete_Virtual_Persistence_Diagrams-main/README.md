# Reproducing Kernel Hilbert Spaces for Non-Discrete Virtual Persistence Diagrams

This repository provides a Python implementation of reproducing kernel Hilbert spaces (RKHS) for virtual persistence diagrams in the non-discrete case, as described in:

**Fanning, C. & Aktas, M. E. (2025).** *Reproducing Kernel Hilbert Spaces for Non-Discrete Virtual Persistence Diagrams.*

## Mathematical Background

### Virtual Persistence Diagrams

Persistent homology associates to a filtered simplicial complex a persistence diagram: a finite multiset of points in $\mathbb{R}^2$ encoding birth and death parameters of homological features. However, finite persistence diagrams form a commutative monoid without additive inverses.

**Virtual persistence diagrams** extend this framework to arbitrary pointed metric spaces. They are constructed by passing to the Grothendieck group $K(X,A)$ of diagrams relative to a metric pair $(X,d,A)$, equipped with the canonical translation-invariant Grothendieck metric $\rho$ extending the Wasserstein-$1$ distance.

### Key Results

1. **Classification**: The metric group $(K(X,A),\rho)$ is locally compact if and only if it is discrete, equivalently when the pointed metric space $(X/A,d_1,[A])$ is uniformly discrete.

2. **Banach-Space Model**: For non-discrete cases, $K(X,A)$ embeds isometrically into its Banach completion $B = \widehat{V}(X,A) \cong \mathcal{F}(X/A,d_1)$, the Lipschitz-free (Arens--Eells) Banach space.

3. **Translation-Invariant Kernels**: Each bounded symmetric positive operator $Q: B \to B^*$ determines a translation-invariant Gaussian kernel:
   $k(x,y) = \exp\left(-\frac{1}{2}\langle Q(x-y), x-y\rangle_{B,B^*}\right)$

4. **Explicit Bounds**: The package provides:
   - Global $\rho$-Lipschitz bounds for all functions in the RKHS
   - Covering-number bounds controlling feature-space complexity
   - Diagrammatic mass bounds from single kernel evaluations
   - Random Fourier feature approximations with probabilistic error control

## Installation

### Requirements

- Python 3.8 or higher
- NumPy, SciPy, Matplotlib
- NetworkX (for graph-based examples)
- ripser, gudhi (for persistence computation)
- scikit-learn (for embeddings)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install Package (Development)

```bash
pip install -e .
```

Or use the modules directly by adding the `src/` directory to your Python path.

## Usage

### Basic Example: Virtual Persistence Diagrams

```python
import numpy as np
from virtual_persistence import MetricPair, PersistenceDiagram, grothendieck_metric

# Create a metric pair (X, d, A)
X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])  # Points in R^2
A = np.array([0])  # Diagonal/basepoint subset

metric_pair = MetricPair(X, A=A)

# Create persistence diagrams
diagram1 = PersistenceDiagram(np.array([[0.5, 1.5], [1.0, 2.0]]))
diagram2 = PersistenceDiagram(np.array([[0.3, 1.2], [1.5, 2.5]]))

# Compute Grothendieck metric
distance = grothendieck_metric(diagram1, diagram2, metric_pair)
print(f"Grothendieck distance: {distance}")
```

### Example: Gaussian RKHS Kernel

```python
from virtual_persistence import (
    HilbertEmbedding, 
    GaussianRKHSKernel,
    GraphLaplacianEmbedding
)

# Create a Hilbert embedding
embedding = GraphLaplacianEmbedding(dim_H=128)
phi = embedding.fit_transform(X)  # Embed points
L = embedding.lipschitz_constant  # Lipschitz constant

# Create kernel
J = HilbertEmbedding(phi, L)
Sigma = np.eye(128)  # Covariance operator
kernel = GaussianRKHSKernel(J, Sigma, t=1.0)

# Evaluate kernel
k_value = kernel(diagram1_embedding, diagram2_embedding)

# Compute Lipschitz bound
f_norm = 1.0
lipschitz_bound = kernel.lipschitz_bound(f_norm)
print(f"Lipschitz bound: {lipschitz_bound}")
```

### Example: Random Fourier Features

```python
from virtual_persistence import RandomFourierFeatures

# Create random Fourier feature approximation
R = 100  # Number of features
rff = RandomFourierFeatures(kernel, R)

# Approximate kernel
k_approx = rff.kernel_approx(diagram1_embedding, diagram2_embedding)

# Concentration bound
epsilon = 0.1
prob_bound = rff.concentration_bound(epsilon)
print(f"P(|k_hat - k| > {epsilon}) <= {prob_bound}")
```

### Example: Persistence Diagram Computation

```python
from virtual_persistence import compute_persistence_diagram

# From point cloud
point_cloud = np.random.randn(100, 3)
diagram = compute_persistence_diagram(point_cloud, method='ripser')

# From graph (using lower-star filtration)
import networkx as nx
G = nx.watts_strogatz_graph(30, 4, 0.3)
graph_data = {'edge_index': np.array(list(G.edges())).T}
diagram = compute_persistence_diagram(
    graph_data, 
    is_graph=True, 
    tau=1.0,
    heat_method='content'
)
```

## Generating Paper Figures

The repository includes scripts to regenerate the figures from the paper:

```bash
# Generate Figures 1-2 (Complex plane and quotient space)
python scripts/figures/paper_figures.py

# Generate Figures 3-4 (Network labelings and dendrograms)
python scripts/figures/petersen_graph_visualization.py
```

Outputs are saved to `results/figures/`.

## Modular Usage

Each theoretical module is independently usable. You can import and use individual components:

```python
# Use only virtual diagrams
from virtual_persistence.theoretical.virtual_diagrams import MetricPair, PersistenceDiagram

# Use only kernels
from virtual_persistence.theoretical.kernels import GaussianRKHSKernel, RandomFourierFeatures

# Use only persistence computation
from virtual_persistence.persistence import compute_persistence_diagram
```

## Theoretical Framework

### Metric Pairs and Virtual Diagrams

A **metric pair** $(X,d,A)$ consists of a metric space $(X,d)$ with a distinguished subset $A \subseteq X$. The $1$-strengthened metric is:
$d_1(x,y) = \min(d(x,y), d(x,A) + d(y,A))$

The **Grothendieck group** $K(X,A)$ extends the monoid of finite persistence diagrams $D(X,A)$ by formally adjoining inverses, with metric:
$\rho(\alpha-\beta, \gamma-\delta) = W_1(\alpha+\delta, \gamma+\beta)$

### Translation-Invariant Kernels

For the Banach completion $B = \widehat{V}(X,A) \cong \mathcal{F}(X/A,d_1)$, each bounded symmetric positive operator $Q: B \to B^*$ determines a Gaussian kernel:
$k_{J,\Sigma,t}(x,y) = \exp\left(-\frac{t}{2}\|\Sigma^{1/2}J(x-y)\|_{\ell^2}^2\right)$

where $J: B \to \ell^2$ is a Hilbert embedding and $\Sigma$ is a trace-class covariance operator.

### Lipschitz Bounds

For every $f \in H_{J,\Sigma,t}$:
$\mathrm{Lip}_\rho(f|_{K(X,A)}) \leq \sqrt{t}\left(\sum_{n\geq 1}\sigma_n w_n^2\right)^{1/2}\|f\|_{H_{J,\Sigma,t}}$

## References

- **Bubenik, P. & Elchesen, A. (2022).** Virtual persistence diagrams, signed measures, Wasserstein distances, and Banach spaces. *Journal of Applied and Computational Topology*, 6, 429--474.

- **Fanning, C. & Aktas, M. E. (2025).** Reproducing Kernel Hilbert Spaces for Non-Discrete Virtual Persistence Diagrams. (Preprint)

## License

MIT License

Copyright (c) 2025 Charles Fanning, Mehmet Emin Aktas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Authors

- Charles Fanning
- Mehmet Emin Aktas

School of Data Science and Analytics, Kennesaw State University
