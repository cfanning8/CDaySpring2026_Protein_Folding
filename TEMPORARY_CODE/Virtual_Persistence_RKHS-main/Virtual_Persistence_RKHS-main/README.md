# Virtual Persistence RKHS Library

RKHS kernels and loss functions for virtual persistence diagrams.

## Mathematical Framework

Virtual persistence diagrams are elements of the Grothendieck group $K(X,A) \cong \mathbb{Z}^{|X\setminus A|}$, represented as integer-valued vectors:

- **Heat Kernel**: Translation-invariant kernel $k_t(\alpha, \beta)$ via heat spectral multipliers
- **Random Fourier Features**: Feature map $\Phi_{t,R}(\alpha)$
- **Topological Loss**: Squared RKHS distance $\|\Phi_{t,R}(\gamma)\|^2$ for VPD differences

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src import TopologicalRKHSLoss, gudhi_persistence_to_vpd_vector

grid_size = 50
loss_fn = TopologicalRKHSLoss(grid_size=grid_size, random_state=14)

vpd = gudhi_persistence_to_vpd_vector(persistence_pairs, grid_size=grid_size, dimension=0)
vpd_diff = vpd_pred - vpd_gt
loss = loss_fn(vpd_diff)
```

See `notebooks/persistent_homology/example.ipynb` for a complete example.

## Library Structure

- `src/kernels.py`: Heat kernel and RFF
- `src/loss.py`: Topological loss function
- `src/vpd.py`: Persistence diagram to VPD conversion

The library does not compute persistence diagrams. Use [gudhi](https://gudhi.inria.fr/).

## Parameters

- `grid_size`: VPD grid dimension (default: 50)
- `n_components`: Number of random features (default: 256)
- `temperature`: Heat kernel temperature (default: 0.2)
- `lambda_weights`: Laplacian edge weights (default: uniform for $d=2$)
