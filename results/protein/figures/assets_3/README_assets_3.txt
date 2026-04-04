assets_3 — publication-style figures (Python stack)

STRUCTURES (structures_py3dmol/)
  py3Dmol + Playwright: ribbon/cartoon with spectrum (native) and b-factor/pLDDT
  coloring (prediction). For journal submission, many groups refine the same
  structures in PyMOL or UCSF ChimeraX (lighting, transparent surfaces, insets).

QUANTITATIVE (quantitative/)
  matplotlib + seaborn: RMSD / pLDDT distributions, scatter, PH summary vs RMSD.

CONTACT MAPS (contact_maps/)
  Cα distance matrices: native, predicted, and absolute difference.

TOPOLOGY / VIRTUAL PD (topology/, virtual_pd/)
  Same 3D persistence engines as elsewhere in this repo (matplotlib).

Requirements: pip install -r requirements-protein.txt and:
  python -m playwright install chromium
