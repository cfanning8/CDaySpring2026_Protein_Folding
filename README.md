# C-DAY: Epidemiology and Protein Folding

## Current status
- Two research tracks share `src/topology/` (persistence, virtual diagrams, RKHS-style losses). That code originated with the epidemiology project; for proteins it is **only** a library for optional **auxiliary loss terms**, not a graph encoder or temporal model.
- Epidemiology: mature TGN + constraint + RKHS scripts under `scripts/`; outputs in `results/output/` and `results/figures/`.
- Protein folding: scaffold under `src/protein/` and `scripts/protein/`; **predictive baseline is ColabFold** (AlphaFold2-class via `colabfold_batch`, not TGN). PDB/mmCIF inputs go under `data/protein/mmcif/` (tracked in this handoff repo). LocalColabFold installs separately from the project venv (see protein smoke section).

## GitHub handoff & automation log (Spring 2026)

This tree is intended to match **[cfanning8/CDaySpring2026_Protein_Folding](https://github.com/cfanning8/CDaySpring2026_Protein_Folding)**. The sections below record what was built in-editor versus what you must run on a GPU machine.

### Implemented in this repository (code + docs)
- **Smoke pipeline:** Layers 1–2 (structure + topology NPZ/tables/figures) and Layer 3 hook for **ColabFold** via `colabfold_batch` (`scripts/protein/run_smoke_pipeline.py`, `scripts/protein/run_colabfold_predict_smoke.py`). Non-ColabFold inference code lives under **`scripts/protein/legacy/`** only (not the publication path).
- **WSL2 workflow:** minimal copy to ext4 (`wsl_sync_minimal_to_home.sh`), project venv (`wsl_setup_protein_venv.sh`), LocalColabFold installer (`wsl_install_localcolabfold.sh`), smoke runner (`wsl_run_smoke.sh` — **syncs `results/protein/` back to the Windows repo by default** via `wsl_sync_results_to_windows.sh`; set `WSL_SKIP_RESULTS_SYNC=1` to skip), env probe (`wsl_probe_env.sh`), torch repair (`wsl_repair_torch_cu124.sh` — **ColabFold default**; legacy OpenFold3 venv repair only if `ENABLE_LEGACY_OPENFOLD3_REPAIR=1`).
- **Guards / probes:** `src/protein/cuda_guard.py` (torch for project venv; `nvidia-smi`-only path for ColabFold subprocess).
- **README:** ColabFold vs huge local DB, `nvcc` vs `nvidia-smi`, WSL symlink / case-sensitivity, low-VRAM flags via `--colabfold-extra-args`.

### Not completed inside the editor (you run on hardware)
- **LocalColabFold install:** long `pixi install && pixi run setup` under `$HOME/localcolabfold` — run `bash scripts/protein/wsl_install_localcolabfold.sh` in **WSL2 on ext4**, not under `/mnt/c/...`.
- **Full Layer 3 inference:** requires `colabfold_batch` on `PATH` and a working **CUDA/JAX** stack; if JAX fails, install the **CUDA toolkit** in WSL so `nvcc --version` aligns with [LocalColabFold](https://github.com/YoshitakaMo/localcolabfold) expectations (driver alone is insufficient for some errors).
- **First** `colabfold_batch` **run** downloads AlphaFold weights (automatic per upstream).

### Clone → end-to-end checklist (condensed)
1. Clone this repo; create a Python **3.10+** venv at repo root if working on Linux/native (optional on Windows for Layers 1–2 only). For protein smoke / mmCIF parsing: **`pip install -r requirements-protein.txt`** (adds **biotite**, py3Dmol, Playwright).
2. **WSL2 (recommended for Layer 3):** `bash scripts/protein/wsl_sync_minimal_to_home.sh` from the Windows-mounted copy, then `cd ~/2_Protein_Folding` (or `WSL_HOME_REPO`).
3. `bash scripts/protein/wsl_setup_protein_venv.sh` → `bash scripts/protein/wsl_install_localcolabfold.sh`.
4. `bash scripts/protein/wsl_run_smoke.sh --skip-structure-figures`  
   Add `--colabfold-extra-args '...'` if VRAM is tight (see smoke section above).  
   After forward sync, `wsl_sync_minimal_to_home.sh` stores your Windows path in `.wsl_windows_repo` on ext4; smoke then **rsyncs `results/protein/` to `/mnt/c/...`** unless `WSL_SKIP_RESULTS_SYNC=1`. Override destination anytime with `WSL_WINDOWS_REPO=/mnt/c/Users/you/.../repo`.

### What is excluded from git (by design)
- Virtualenvs: `.venv/`, `.venv-wsl-gpu/`, etc.
- Cursor local files: `.cursorrules`, `.cursor/`
- Large regenerated caches: `data/processed/protein/*.npz`
- Everything else in this handoff—including **smoke mmCIF under `data/protein/mmcif/`** and **results under `results/protein/`** for figures/tables—is meant to be versioned unless a file exceeds GitHub’s size limits; split or Git LFS if a future artifact is too large.

## Repository map
- `src/` : `topology/` (shared math for persistence and losses), `models/` (**epidemiology only**: TGN stack), epidemiology loaders and simulators, `protein/` (**ColabFold-aligned** corpus + topology preprocessing for **loss-side** geometry; no TGN; `src/protein/legacy/` holds embargoed non-ColabFold shims only).
- `scripts/` : epidemiology entry points at top level; protein folding under `scripts/protein/`.
- `data/epidemiology/` : optional home for epi inputs separate from legacy `data/preprocessed/` paths used by existing scripts.
- `data/protein/mmcif/` : deposited entry mmCIF files (not biological assemblies; tracked in this handoff repo).
- `data/protein/validation/` : validation PDFs paired to entries (gitignored blobs).
- `data/processed/protein/` : cached topology NPZs (coords, edge table, persistence arrays; `*.npz` gitignored).
- `results/output/` and `results/figures/` : epidemiology runs.
- `results/protein/output/` and `results/protein/figures/` : protein runs.
- `requirements-base.txt` : shared numerics/topology stack.
- `requirements-epi.txt` : default install (includes PyG and PyVista).
- `requirements-protein.txt` : Biotite, py3Dmol, Playwright for structure figures; no PyG/TGN stack. Fine-tuning uses **ColabFold/JAX** outside this venv; optional `requirements-openfold3-smoke.txt` is **legacy only** (`scripts/protein/legacy/`).

## Python environment
- Create a venv in the repo root, then install the track you need:
  - Epidemiology (default): `pip install -r requirements.txt` (same as `requirements-epi.txt`).
  - Protein helpers only: `pip install -r requirements-protein.txt`.
  - Both in one env: `pip install -r requirements-epi.txt` then `pip install -r requirements-protein.txt` (shared base deduped by pip).

## Epidemiology track

### Frozen research claim
- Main: virtual topological drift predicts epidemic instability.
- Supporting: topology-aware models beat graph-only temporal baselines.

### Frozen model set
- Model 1: TGN only (`scripts/train_tgn_baseline.py`).
- Model 2: TGN + PersLay + constraint (`scripts/train_tgn_perslay_constraint.py`).
- Model 3: TGN + RKHS + constraint (`scripts/train_tgn_rkhs_constraint.py`).

### Frozen target and simulator policy
- Target `Y_t = P(attack_rate >= tau)` with `tau = 0.20`, SIR simulator, per-dataset calibration frozen in `results/output/sir_calibration_by_dataset.csv`.
- Chronological splits for train/val/test (no random window shuffling for official eval).

### Key modules and scripts
- Loaders and windows: `src/dataloaders.py`, `src/window_cache.py`, `src/edge_preparation.py`, `src/episim.py`.
- Topology feature cache: `scripts/cache_persistence_features.py`.
- Collective benchmark driver: `scripts/run_collective_benchmark.py`.
- Topology smoke test (shared math): `scripts/smoke_test_paper1_tools.py`.
- Figures: `scripts/figures/generate_project_assets.py` and outputs under `results/figures/`.

### Frozen calibration and benchmark commands (epi)
- Full SIR calibration (writes `results/output/sir_calibration_by_dataset.csv`):  
  `python -u scripts/calibrate_sir_all_datasets.py --sample-windows 6 --calibration-num-simulations 10 --search-rounds 1 --beta-scales 0.7,1.0 --gamma-scales 1.0,1.4 --apply-full-cache --full-num-simulations 120 --full-workers 8 --full-flush-every 2`
- One-dataset rebuild + train:  
  `python -u scripts/run_collective_benchmark.py --dataset <DATASET> --reset-cache --force-rebuild --force-train`
- Topology loss tuning (offline):  
  `python -u scripts/tune_topology_loss_terms.py --dataset <DATASET>`

### Evaluation
- Primary: RMSE on chronological test; secondary: Brier, ECE.

## Protein folding track

### Design freeze and alignment audit (2026-04; authoritative)

This block is a **specification**, not background reading.

**Interpretation rule.** If repository behavior or older README prose **differs** from a statement here, treat that as a **defect** until a deliberate amendment is recorded. “Close enough” is not sufficient for the protein track: stage semantics, loss identity, domain units, and numerical parity with the **Virtual Persistence RKHS** reference code path are **material**.

**Severity scale**

| Tag | Meaning |
|-----|---------|
| **S0** | **Blocking** for scientific or numerical claims: wrong training semantics, wrong parity with the paper reference implementation, or topology framed as input/diagnostic/failure-prediction. |
| **S1** | **Protocol / unit-of-analysis**: splits, CATH alignment, residue register, benchmark versus smoke policies. |
| **S2** | **Engineering / narrative hygiene**: naming, layout, smoke-only placeholders that must not be mistaken for the target estimand. |

The bullet list below records **binding PI intent**. The **Exhaustive file–level defect registry** lists **every** protein-adjacent path audited in-editor and what must change.

#### Frozen scientific identity

1. **Primary claim:** **Fine-tune** a **local ColabFold / AlphaFold2-weighted** model so that **additional topological loss terms** change **coordinate predictions** under an **augmented objective**. This is **not** a topology-as-input architecture, **not** a post-hoc structural diagnostic as the main scientific story, **not** failure-from-topology prediction, and **not** PersLay-style representation feeding the trunk.
2. **Baseline inference stack for narrative:** **ColabFold only** for the intended protein publication / training story. Third-party AF3-class installers and drivers are **quarantined** under `scripts/protein/legacy/` + `src/protein/legacy/` and **must not** appear in methodology prose.
3. **Stage axis (three mutually exclusive training recipes — “competing,” not cumulative):**
   - **Stage 0 — Baseline:** standard folding loss only (AF2-class / ColabFold training objective as implemented in the fine-tuning stack).
   - **Stage 1 — Wasserstein:** same baseline **plus** a **Wasserstein(-type) topological loss** on persistence objects at the relevant **structural depth** (see depth axis below). **Not** “persistence landscapes” as a named stage; **eliminate persistence-landscape language** for protein staging.
   - **Stage 2 — RKHS:** same baseline **plus** \(L_{\mathcal{H}}\): the **RKHS semimetric** on **virtual persistence** built from **CATH-domain internal** comparisons with **sum pooling** \(\Gamma=\sum_{(a,b)\in\mathcal{C}(P)}(D_a^{(n)}-D_b^{(n)})\) (README **Full differential pipeline**). **Formal analysis** uses **characters, heat measure, and spectral symbol** on that virtual object; **`src/topology/`** implements a **grid encoder + heat-kernel RFF** surrogate aligned with `TEMPORARY_CODE/Virtual_Persistence_RKHS-main/Virtual_Persistence_RKHS-main/` (**Reference package vs `src/topology`**).
   - **Do not stack** Stage 1 and Stage 2 in one training run for the primary experimental comparison; each stage is a **separate trained model** (or separate run) compared **against** Baseline.
4. **Depth axis (orthogonal to stage — Cartesian product):** “Higher-order” objects (**interval-of-intervals**, recursive lifts \(D^{(n)}\)) apply to **both** the Wasserstein arm and the RKHS arm. Depth is **not** “later stages”; it indexes **which monoid lift** you compare. Baseline **does not** use this machinery. Detailed combinatorics are defined in the attached mathematical note (not duplicated here).
5. **CATH roles (both):** (i) **Train/test binning** by superfamily (and optional S35) as already documented. (ii) **Domain-induced modules inside one protein**: multiple **comparable** coordinate substructures so that **internal** persistence diagrams exist in parallel; **virtual persistence diagrams** arise from **controlled differences among those internal objects** (and the same construction applied to **predicted** geometry). Smoke’s **three contiguous thirds** are **not** this; they are **temporary plumbing** until domain-boundary parsing lands.
6. **Virtual persistence (concept):** Differences live in a space of **diagrams with signed multiplicities** (Grothendieck-completed / monoid picture per your note). **Binning onto a fixed grid** is an **implementation encoding** for feeding **RFF**, not the definition of a VPD.
7. **Evaluation endpoints:** **TM-score** and **GDT-TS** are **primary** reporting targets for success vs baseline. **CA RMSD** and **pLDDT** are **secondary / dev**. **Topological losses** are **optimization terms** reported as part of training—not mistaken for primary benchmarks, and **not** framed as throwaway “ablation trivia.”
8. **Minimum experiment that matches intent:** Non-smoke **CATH-based** corpus, **ColabFold fine-tuning**, **three competing stage recipes**, **CATH-domain modules** (not contiguous splits), **depth-\(n\)** variants as the note specifies, evaluation on **TM-score, GDT-TS, RMSD, pLDDT**.

#### Terminology law (binding)

| Term | Meaning | Forbidden misuse |
|------|---------|------------------|
| **Virtual persistence diagram (VPD)** | Diagram data with **signed integer multiplicities** on cells (or formal monoid/Grothendieck image), obtained by **legal difference** operations on ordinary diagrams from **comparable** substructures. | Calling any raw finite set of `(birth, death)` intervals “the VPD” without going through the difference/construction. |
| **Ordinary / module diagram** | Persistence diagram \(D(P_a)\) from the frozen filtration on coordinates restricted to module \(P_a\). | Equating this with the training-stage name “Baseline”; baseline is **only** the AF2-class loss recipe. |
| **Grid / occupancy encoder** | Fixed `grid_size²` integer (or embedded) vector produced by binning mass from an ordinary diagram (implementation in `gudhi_persistence_to_vpd_vector`). | Describing this encoder output as if it **were** the VPD **object** in the paper sense; it is **`Enc(D)`**, a finite encoding chosen so `HeatRandomFeatures` has fixed `input_dim`. |
| **Training stage** | One of **Baseline**, **Wasserstein**, **RKHS** as a **competing** full training recipe (\(L = L_{\mathrm{fold}}\) or \(L = L_{\mathrm{fold}} + \lambda_{\mathrm{W}} L_{\mathrm{W}}\) or \(L = L_{\mathrm{fold}} + \lambda_{\mathcal{H}} L_{\mathcal{H}}\)), **not** cumulative stacking of Wasserstein + RKHS for primary comparisons. | “Landscape”; “additive stages” in one run when the PI intent is **separate trained models**; **persistence landscape** as stage name. |
| **Structural depth \(n\)** | Ordinal lift (interval-of-intervals, recursive monoid) applied **inside** a stage **and** applicable to **both** Wasserstein-based and RKHS-based comparisons at that depth. | Treating depth as a third “stage” or as RKHS-only. |
| **Primary metrics** | **TM-score**, **GDT-TS** for publication success. | Reporting Wasserstein / RKHS as headline “metrics” rather than **optimization / training losses**. |
| **Secondary / dev** | **CA RMSD**, **pLDDT** (and calibration-style summaries if reported honestly). | Using smoke truncation alignment for benchmark tables. |

Higher-order containment rules and birth–death assignment for \(D^{(n)}\) are **defined only in the attached mathematics file** referenced by the PI; until that file lives in-repo next to tests, **DEF-M01 [S0]** applies: **no code in this repository implements the monoid lift as specified there**.

#### Exhaustive file–level defect registry (protein + topology)

Each line is an auditable item. **Status** is `open` until closed in git with code/doc change.

**Shared topology (loss path)**

| ID | Sev | Path | Defect | Required closure |
|----|-----|------|--------|------------------|
| DEF-K01 | S0 | `src/topology/kernels.py` | ~~`laplacian_symbol` lacked reference 2-torus branch~~ | **Closed:** matches `TEMPORARY_CODE/.../kernels.py` (add regression test). |
| DEF-K02 | S0 | `src/topology/kernels.py` | ~~`default_rng` vs `check_random_state`~~ | **Closed:** `check_random_state`; **expect seed output drift** vs any older runs. |
| DEF-K03 | S0 | `src/topology/kernels.py` | ~~`effective_temp` divisor~~ | **Closed:** matches reference `input_dim > 2` branch. |
| DEF-L01 | S0 | `src/topology/loss.py` | ~~Missing `topological_loss_batch_torch` export~~ | **Closed:** exported from `loss.py` and `src/topology/__init__.py`. |
| DEF-L02 | S0 | `src/topology/loss.py` | `topological_loss_batch_torch` **detaches** (NumPy loop). | **Partial:** `TopologicalRKHSLossTorch` is **fully differentiable w.r.t. the diagram encoding**; the **intended** \(\widehat{X}\!\to\!\text{PH}\!\to\!\omega\) chain is **piecewise smooth on fixed strata** (README **Full differential pipeline**). Repo smoke still uses **Gudhi → NumPy encoder** without stratum-tracking autograd. |
| DEF-V01 | S2 | `src/topology/vpd.py` | Encoder functions named `*vpd_vector*` encourage conflation of **`Enc(D)`** with VPD (**see Terminology law**). | Rename or document in-module only; prose uses “encoder of diagram mass.” |
| DEF-V02 | S1 | `src/topology/vpd.py` | Binning uses `searchsorted`; finite `inf` surrogate `death = birth + 1.0` — changes diagram; must be **frozen** and versioned in NPZ if cited. | Document in selection spec; optional flag for `inf` handling. |
| DEF-P01 | S2 | `src/topology/perslay_module.py` | **Epidemiology only**; must never be imported by protein training. | Grep-gate / CI: protein tree must not import. |

**Protein corpus and metadata**

| ID | Sev | Path | Defect | Required closure |
|----|-----|------|--------|------------------|
| DEF-D01 | S1 | `src/protein/dataset_policy.py` | Phase-1 **whole-chain** CATH domains only; **chopped** domains and **multi-domain-per-chain** coordinate windows **not** Frozen for training modules. | Boundary parser + policy revision or explicit phase label in all tables. |
| DEF-D02 | S1 | `src/protein/eligible_domains.py` + manifests | “Domain” row does not yet guarantee **coordinate subset** used in PH matches **domain residue ranges** for multi-domain physics. | Join CATH ranges to mmCIF residue index map; tests. |
| DEF-M01 | S1 | `src/protein/mmcif_io.py` | Chain-centric extraction; domain trimming for PH not applied here. | Domain slice helper used by PH + ColabFold inputs consistently. |
| DEF-R01 | S2 | `src/protein/rcsb_data_api.py`, `pdb_manifest_schema.py`, `pdb_strata.py` | **Strong** for metadata; **no defect** on intent — but downstream **must** consume `entry_exclusion_code` and stratum for reporting. | Document attrition tables for paper. |

**Protein topology geometry**

| ID | Sev | Path | Defect | Required closure |
|----|-----|------|--------|------------------|
| DEF-G01 | S1 | `src/protein/residue_points.py`, `residue_graph.py`, `clique_persistence.py` | Representative choice (Cβ virtual), **8 Å**, backbone **0** filtration — **policy constants**, not yet sensitivity study. | Freeze in supplemental; ablation plan. |
| DEF-G02 | S1 | `src/protein/topology_cache.py` | NPZ stores **ordinary** persistence only; **no** \(D^{(n)}\) cache policy decided. | Spec: on-the-fly vs cache; version key bump. |

**Protein smoke and diagnostics (must not be mistaken for training)**

| ID | Sev | Path | Defect | Required closure |
|----|-----|------|--------|------------------|
| DEF-S01 | S0 | `src/protein/smoke_nine_experiments.py` | Uses **contiguous thirds**, not CATH modules; RKHS path uses **pred−native** diagram difference per third, **not** internal \(\mathcal{C}(P)\) VPD family on **both** \(\widehat P\) and \(P^\ast\) as specified. | Rewrite against domain slices + internal pair family; or label output `smoke_plumbing_v1` only. |
| DEF-S02 | S2 | `src/protein/smoke_nine_experiments.py`, `smoke_metrics.py`, `smoke_table_render.py`, `run_smoke_pipeline.py` | Stage label **Landscape** for **\(W_2\)**. | **Closed** in source: **`Wasserstein`**. Regenerate `results/protein/tables/smoke/*.csv` + PNG to refresh committed artifacts. |
| DEF-S03 | S1 | `smoke_metrics.py` | `min(len)` alignment masks register errors — correct for **smoke only** [S2 if leaked]. | Banner in CSV; reject in benchmark driver. |
| DEF-S04 | S2 | `smoke_table_render.py` | Single column **RMSD** mixed scales. | **Closed:** PNG uses **TM-score** (TM-align when `TMalign` on `PATH` or `TMALIGN_BIN`; else Zhang–Skolnick fallback) + **pLDDT**. Install: `scripts/protein/install_tm_align_linux.sh`. |
| DEF-S05 | S0 | `scripts/protein/run_smoke_pipeline.py` | **Inference-only** ColabFold; no fine-tuning loop, no Wasserstein/RKHS **training** attachment. | New training package per PI spec. |

**Out-of-scope / embargoed narrative (protein publication)**

| ID | Sev | Path | Defect | Required closure |
|----|-----|------|--------|------------------|
| DEF-O01 | S0 | Legacy OpenFold3 stack | Was on publication-adjacent paths. | **Closed in tree:** `scripts/protein/legacy/*`, `src/protein/legacy/openfold3_query.py`, shim `src/protein/openfold3_query.py`; stubs forward from old script paths; `wsl_repair` defaults **omit** OpenFold3 pip unless `ENABLE_LEGACY_OPENFOLD3_REPAIR=1`. |
| DEF-O02 | S0 | README contradictions | Landscape / cumulative / OpenFold happy-path. | **Closed** for protein narrative + competing stages + derivation block; **grep periodically** so handoff edits do not regress. |

**Scripts layout (`scripts/protein/`)**

| ID | Sev | Path | Defect | Required closure |
|----|-----|------|--------|------------------|
| DEF-J01 | S2 | `run_smoke_pipeline.py` | Large orchestration logic in `scripts/` — harder to unit test than `src/protein/...` module. | Extract `SmokePipelineRunner` (or equivalent) into `src/protein` when stabilizing. |
| DEF-J02 | S2 | `smoke_persistence_3d.py` | Lives under `scripts/` but behaves like a library module (imported). | Move to `src/protein` or `src/figures` with clear package boundary. |
| DEF-J03 | S2 | WSL scripts | Mixed messaging. | **Closed:** `wsl_git_clone`, `wsl_repair`, `wsl_run_smoke` head comments ColabFold-first; legacy isolated. |

#### README sections that still contradict the freeze (inventory)

| Location | Contradiction | Severity |
|----------|---------------|----------|
| **Axes: learning stages** | Old “additive” three-in-one | **Closed** (competing recipes). |
| **Formal theory §9** | Old stacked sum with \(L_{\mathrm{land}}\) | **Closed** (three \(L^{(i)}\)). |
| **Planned model ladder** | Landscape / jet wording | **Closed** (Wasserstein + depth grid). |
| **Layers → Learning** | Landscape phrase | **Closed**. |
| **Reconstruction guide §1** | persistence landscape | **Closed**. |
| **Repository map** | OpenFold-facing | **Closed** (ColabFold + legacy note). |

These rows are closed only when the prose is rewritten **or** explicitly marked `LEGACY—see Design freeze` with no conflicting claim.

#### Reference package vs `src/topology` (reconciliation required)

The folder `TEMPORARY_CODE/Virtual_Persistence_RKHS-main/Virtual_Persistence_RKHS-main/src/` is the **paper-aligned reference** for kernels + RKHS loss + VPD vectorization. `src/topology/` is the **integrated** copy used by the epidemiology and protein scaffolding. A line-by-line diff matters for **reproducibility claims**:

| Item | Reference (`TEMPORARY_CODE/...`) | Integrated (`src/topology/`) | Status |
|------|-----------------------------------|------------------------------|--------|
| `laplacian_symbol` | Anisotropic 2-torus branch + weighted tensordot path | **Matched** in `kernels.py` (same branches and formulas) | **DEF-K01 — closed** |
| RNG | `sklearn.utils.check_random_state` | **Matched** | **DEF-K02 — closed** |
| `effective_temp` | `temperature / input_dim` if `input_dim > 2` else `temperature` | **Matched** | **DEF-K03 — closed** |
| `TopologicalRKHSLoss` | `np.sum(self._zero_embed ** 2)` vs dot | Dot product (equal) | No action |
| Finity guard on loss | absent | `if not np.isfinite: return 0.0` | **Open** — document or drop for strict parity |
| `topological_loss_batch_torch` | Present; **not autograd** | **Exported**; same non-differentiable pattern | **DEF-L01 — closed**; **DEF-L02 — partial** (`TopologicalRKHSLossTorch` for encoding grad) |
| Import paths | `from src.kernels import ...` | `from src.topology.kernels import ...` | Hygiene only |

**Regression expectation:** Epidemiology or protein code that **depended** on the previous `default_rng` draws will observe **different** RFF weights at the same integer seed — this is **intentional** alignment with the paper reference package, not a bug.

**Claims:** With **DEF-K01–K03** and **DEF-L01** closed, fixed-seed comparisons to `TEMPORARY_CODE` notebooks are **possible** subject to remaining guards (`TopologicalRKHSLoss` finity), dtype paths, and `vpd` encoder parity. The **intended** end-to-end gradient \(\partial L/\partial\omega\) through **filtration values \(\to\)** birth/death \(\to\) transport \(\to\) Wasserstein / \(L_{\mathcal{H}}\) is **specified piecewise-smoothly** in README **Full differential pipeline** (Clarke calculus at breakpoints). **`TopologicalRKHSLossTorch`** in-repo demonstrates full gradients **through the diagram mass encoding** only (see `src/protein/training_contract.py`, `scripts/protein/run_training_contract_demo.py`); wiring that encoding to a **differentiable PH stratum** in JAX/ColabFold training remains **DEF-S05** work.

#### Gap catalog (implementation and documentation vs freeze)

**A. Narrative / README contradictions (fix in prose first, then code)**

1. **“Landscape” stage name:** **Addressed** (**DEF-S02**); no `Landscape` in protein **source** (`verify_protein_alignment.py` enforces); regenerate committed smoke CSV/PNG if headers still say **Landscape**.
2. **Cumulative objective in formal §9:** **Replaced** with **three competing** objectives \(L^{(0)}, L^{(1)}, L^{(2)}\) in the formal block below; sweep any remaining “additive across stages” language in older handoff bullets.
3. **“Persistence landscape + RKHS”** — reconstruction guide §1 **updated**; grep README for stray “persistence landscape” in protein context.
4. **“Precomputed native features”** (opening paragraph of this track): misleading if read as **native topology as model input**; native enters as **targets / reference diagrams** for losses, **not** trunk features.
5. **OpenFold / OpenFold3** — **sandboxed** (`scripts/protein/legacy/`, `src/protein/legacy/`); stubs and README default to **ColabFold-only**.
6. **Depth vs stage:** Older text ties “interval-of-intervals” only to RKHS or “later jet”; PI: **same depth axis applies to Wasserstein comparisons** of lifted objects.
7. **CATH “only for bins”** undertone: README must state clearly that domains also define **internal module families** for **VPD algebra**.

**B. Smoke / diagnostics vs training estimand**

8. **No ColabFold / JAX fine-tuning entry point:** Smoke is **inference-only**; **`src/protein/training_contract.py`** + **`scripts/protein/run_training_contract_demo.py`** document the **PyTorch RKHS** arm and prove **encoding gradients**. Full AF2 objective coupling remains **out of repo** (**DEF-S05** partial).
9. **`smoke_nine_experiments` module model:** **Contiguous `array_split`** thirds + **PH on isolated CA subset** — PI: **temporary**; target is **CATH domain boundaries** (and later chopped domains when boundary parser exists).
10. **VPD in smoke:** Implements **pred-vs-native** diagram difference **per contiguous third** — PI training story emphasizes **internal multi-domain (multi-substructure) comparisons** **within** native and **within** predicted structures, then **RKHS between** aggregated virtual objects (and Wasserstein on ordinary or lifted diagrams as appropriate). Smoke **does not** instantiate the internal-comparison family \(\mathcal{C}(P)\).
11. **Nine-cell PNG table:** ~~mixed “RMSD” column~~ — **fixed** (**DEF-S04**); columns **TM-score \| pLDDT**; CSV one numeric slot per metric column per row.
12. **`compute_per_structure_metrics` truncation:** `min(len_native, len_pred)` — acceptable for smoke; **unacceptable** as benchmark alignment policy when sequence/register errors exist.

**C. Data pipeline vs “CATH domain module” claim**

13. **Phase-1 policy:** Whole-chain CATH domains only; **chopped** domains deferred — blocks full “domain = module” until boundary pass lands.
14. **No multi-domain coordinate extraction** in one mmCIF row for **side-by-side** PH on each domain instance in the same chain (as needed for internal VPD family).

**D. Higher-order / monoid lift**

15. **\(D^{(n)}\) lifts:** **Not implemented** in Python pipeline; only described in theory. Smoke explicitly skips.
16. **Wasserstein on lifted objects:** No code.

**E. Evaluation stack**

17. **TM-score:** **Smoke** uses `src/protein/tm_score_eval.py` (TM-align primary; CASP-comparable). **GDT-TS:** not wired into `results/protein` evaluation scripts yet.

**F. Epidemiology bleed**

18. **`src/topology/perslay_module.py`:** Must never appear in **protein** training narrative (epi only).
19. **Alignment losses** (`pairwise_alignment_loss`, etc.): PI forbids **feature-space topology alignment** as a substitute for the **loss-term** design; ensure protein docs don’t imply that path.

#### Open design knobs (still need PI lock after math note is coded)

- **Exact family \(\mathcal{C}(P)\)** of internal CATH-domain-based comparable pairs that generate **multiple VPDs** per protein.
- **RKHS pooling** over internal VPDs for one protein-level scalar: **frozen as sum pooling** \(\Gamma=\sum_{(a,b)\in\mathcal{C}(P)} G_{a,b}^{(n)}\) (see **Full differential pipeline** §7 below). Other poolings remain **out of primary scope** unless the experiment grid explicitly reopens them.
- **Wasserstein vs RKHS roles** at each depth \(n\): **pred vs native** \(\widehat{X}\) vs \(X^\ast\) for **Wasserstein**; **internal virtual differences** + **sum-pooled** RKHS semimetric for **RKHS** — formal objectives in §6–§7 below.
- **Cache policy:** whether lifted objects \(D^{(n)}\) are **precomputed in NPZ** or **built on the fly** during training.
- **ColabFold / JAX engineering path** for true \(\partial L/\partial \omega\) through the AF2 stack (when PyTorch surrogates are insufficient) remains an implementation audit item.

#### Planned overhaul phases (documentation checklist; implementation TBD)

| Phase | Scope |
|-------|--------|
| P0 | Rewrite protein-track README sections and smoke captions: **Wasserstein** not Landscape; **competing stages**; **ColabFold-only** narrative; **no PersLay / OpenFold / landscape / failure-prediction** in protein story; notation \(L_{\mathcal{H}}\), sum pooling, derivation block. (**Closed** in authoritative README blocks; smoke **artifacts** may lag until rerun.) |
| P1 | Reconcile `src/topology/kernels.py` + `loss.py` with `TEMPORARY_CODE/Virtual_Persistence_RKHS-main` (+ unit tests vs reference notebook outputs). (**Kernel/RNG/temperature + torch batch export landed**; add regression tests.) |
| P2 | Replace smoke module axis with **CATH-domain slices** when data path exists; implement internal \(\mathcal{C}(P)\) builder. |
| P3 | Implement **Wasserstein training loss** path at depth 0 (then lifted depths). |
| P4 | Implement **RKHS** path with **multiple VPDs** + aggregation; torch autograd story. |
| P5 | Benchmark evaluation: **TM-score**, **GDT-TS**, secondary RMSD/pLDDT; frozen splits. |

---

This track is **ColabFold / AlphaFold2-class structure prediction + cached clique-persistence topology**, not “a protein TGN.” The epidemiology project is a **historical analogy** only (both projects ask when topology-informed objectives help); the protein **structure baseline** is **ColabFold** (`colabfold_batch`), with topology entering as **targets and constraints in the loss** (reference/native geometry for diagram comparisons), **not** as features concatenated into the folding trunk. The target is **additive topological loss terms** during **fine-tuning** on top of the standard folding objective (**see Design freeze** for competing-stage semantics).

### Layers (frozen conceptually)
1. **Structural data:** mmCIF parsing, chain/domain choice, residue representatives (see topology policy), metadata, manifests, caching keys (`mmcif` hash + topology policy id).
2. **Topology preprocessing:** weighted residue graph, clique filtration, persistence, NPZ cache under `data/processed/protein/`. This graph is a **derived geometric object for PH**; it is **not** the ColabFold/AlphaFold2 model graph.
3. **Learning:** **ColabFold** + **fine-tuning** target: **ColabFold** inference (public MSA server by default for small jobs) on the eligible manifest; **training** augments \(L_{\mathrm{fold}}\) with **either** a **Wasserstein** topological loss **or** an **RKHS** loss on **virtual persistence diagrams** (**competing** recipes, not stacked for primary comparisons). **Structural depth** \(n\) is a **second axis** (Cartesian with stage) for **both** Wasserstein and RKHS constructions. Attachment is **loss-side on predicted coordinates** (and reference native for targets), **not** topology fed into the trunk.

### Scientific question (prediction estimand)
- Do **competing fine-tuning recipes** (Baseline vs **+Wasserstein** vs **+RKHS**), holding the **ColabFold / AF2-class** core, improve **TM-score / GDT-TS** under hard generalization (held-out CATH homologous superfamilies)?

### Axes: learning stages vs structural depth vs pipeline layers
- **Pipeline layers** (implementation): mmCIF → topology cache → learning. These are **not** the same as the two conceptual axes below.
- **Learning stages** (**three competing training recipes** — train **separate** models or runs; do **not** stack Wasserstein + RKHS losses for the primary experimental comparison):
  1. **Baseline** — \(L = L_{\mathrm{fold}}\) only (ColabFold / AF2-class; current smoke runs **inference** only — **DEF-S05**).
  2. **Wasserstein** — \(L = L_{\mathrm{fold}} + \lambda_{\mathrm{W}} L_{\mathrm{W}}\) with \(L_{\mathrm{W}}\) a **Wasserstein / diagram comparison** loss at the chosen **structural depth** (ordinary or lifted diagrams per the math note).
  3. **RKHS** — \(L = L_{\mathrm{fold}} + \lambda_{\mathcal{H}} L_{\mathcal{H}}\) with \(L_{\mathcal{H}}\) the **RKHS semimetric** on **sum-pooled** internal virtual objects (see **Full differential pipeline**). **`src/topology/`** supplies `HeatRandomFeatures` + `TopologicalRKHSLoss` as **RFF encoding** surrogate (**DEF-K01–K03**). Persistence **landscape** terminology is **retired** for this axis.
- **Structural depth** (ordinal \(n=0,1,2,\ldots\)): indexes **which monoid-lifted object** is compared — **orthogonal to stage**: **both** Wasserstein and RKHS stages may compare at depth \(n\). Baseline does **not** use this machinery.
- **Smoke table (legacy plumbing):** nine cells use **contiguous thirds** × **Baseline / Wasserstein / RKHS**. PNG reports **TM-score** + **pLDDT** (**DEF-S04** closed). Until separate stage checkpoints exist, one ColabFold prediction yields the **same** TM-score across stages at a given level (honest labeling). Rows remain **diagnostics only** — wrong module semantics (**DEF-S01**).

### Formal theory: protein-internal topological drift (modules, virtual diagrams, RKHS)
Protein folding takes a sequence and predicts a 3D structure. Baseline losses are standard; we **add** topology-informed terms **without** concatenating topology into the folding trunk. There is **no natural time axis** for a single static structure, so “drift” cannot mean temporal change. The replacement is **internal hierarchical topological drift**: structured variation across **comparable substructures inside one protein**, combined with **higher-level persistence aggregation** so that drift is meaningful beyond first-order diagrams.

**Core idea (two ingredients).** (i) **Internal modular comparison** produces **formal differences** of diagrams, hence **virtual persistence diagrams** (signed multiplicity, pointwise subtraction in the Grothendieck completion) — the algebraic input required for RKHS kernels. (ii) **Higher-level interval aggregation** produces **structural depth**: nested or co-occurring intervals become new diagram-level atoms (interval-of-intervals, recursive monoid lifts as in your deeper notes). Together:

\[
\text{module comparison} + \text{structural depth}.
\]

#### 1. Protein as a family of comparable substructures
Represent a structure as a finite set \(P=\{x_1,\dots,x_N\}\subset\mathbb{R}^3\) (atoms/residues with labels as needed). Choose modules \(\mathcal{M}(P)=\{P_a:a\in A\}\), \(P_a\subseteq P\), from domains, secondary-structure blocks, windows, contact-graph neighborhoods, nested coarse-to-fine decompositions, etc. Prefer a partial order \(\sqsubseteq\) with \(P_b\sqsubseteq P_a\) meaning “contained in” (inclusion, domain containment, neighborhood containment, or scale containment). This **ordered module family replaces epidemic time windows**: comparable internal states instead of comparable times.

#### 2. Ordinary persistence on each module
For each module \(P_a\), build a filtration (VR, alpha, weighted contact, …) and obtain a persistence diagram \(D_a^{(1)}:=D(P_a)\). The protein induces a **diagram field** \(a\mapsto D_a^{(1)}\).

#### 3. Internal virtual persistence diagrams from module comparison
Fix comparable pairs \(\mathcal{C}(P)\subseteq A\times A\); a principled choice is \((a,b)\) with \(P_b\sqsubseteq P_a\), \(a\neq b\). Define the **internal virtual diagram** at level 1:

\[
G_{a,b}^{(1)} := D_a^{(1)} - D_b^{(1)}.
\]

This is **intra-structural drift**: signed mass records features gained or lost from the smaller module to the larger. It is the protein analogue of drift **without** time series.

#### 4. Why this suffices for RKHS machinery
Virtual diagrams live in the Grothendieck completion \(K^{(1)}(X)=K(D^{(1)}(X))\), and at depth \(n\), \(K^{(n)}(X)=K(D^{(n)}(X))\). Kernels, feature maps, and Bochner-type factorizations require **signed** objects; modular comparison supplies them.

#### 5. Higher-level aggregation (structural depth)
Write \(D_a^{(1)}=\sum_i m_i I_i\) for intervals \(I_i\) with multiplicities. Preorder intervals by containment \(I_j \preceq_{\mathrm{int}} I_i\) iff \(I_i\) contains \(I_j\). A **level-2** aggregate keeps co-occurring nested pairs:

\[
D_a^{(2)} := \sum_{i,j:\, I_j \preceq_{\mathrm{int}} I_i} m_i m_j \,(I_j,I_i).
\]

Iterate (or use the full recursive monoid construction \(D^{(n+1)}(X)=D(X^{(n)},A^{(n)})\) from your notes). This supplies **second-order structure** (relations among features), not only isolated intervals.

#### 6. Why aggregation matters in addition to module comparison
Module differences \(G_{a,b}^{(1)}=D_a^{(1)}-D_b^{(1)}\) compare first-order features; they do **not** encode whether **relations among features** differ. Aggregation to \(D^{(n)}\) for \(n\ge 2\) captures nested relations (co-occurring loops, cavities-within-domains, etc.).

#### 7. Level-\(n\) internal drift field
Given modules, comparability \(\mathcal{C}(P)\), and hierarchy \(D_a^{(n)}\), define

\[
\mathcal{G}^{(n)}(P) = \left\{ G_{a,b}^{(n)} := D_a^{(n)} - D_b^{(n)} \mid (a,b)\in\mathcal{C}(P)\right\}.
\]

At \(n=1\), ordinary module diagrams; at \(n=2\), interval-of-interval objects; higher \(n\) compares recursively lifted structures. **Drift** means **change across ordered internal structure at topological depth**, not change over time.

#### 8. Pooling to a protein-level RKHS object (sum pooling — frozen)
Let operations live in \(K^{(n)}(X)\) before the RKHS feature map. **Primary training spec:** **sum** the internal virtual objects,

\[
\Gamma_P^{(n)} = \sum_{(a,b)\in\mathcal{C}(P)} \bigl( D_a^{(n)} - D_b^{(n)} \bigr) = \sum_{(a,b)\in\mathcal{C}(P)} G_{a,b}^{(n)},
\]

then apply \(\Phi_t\) (paper RKHS map) to \(\Gamma_{\widehat{X}}^{(n)}\) and \(\Gamma_{X^\ast}^{(n)}\). **Mean** or attention-weighted pooling is **not** the frozen definition for publication-stage RKHS loss (unless a separate ablation column is added).

#### 9. Combined learning objective (competing trainings, not one stacked sum)
Let \(\widehat{P}\) be the prediction and \(P^\ast\) the native structure. **Primary publication comparison:** three **separate** trained models (or runs), not one objective with all penalties turned on:

\[
\begin{aligned}
\text{(Baseline)} \quad & L^{(0)} = L_{\mathrm{fold}}(\widehat{P}, P^\ast), \\
\text{(Wasserstein)} \quad & L^{(1)} = L_{\mathrm{fold}}(\widehat{P}, P^\ast) + \lambda_{\mathrm{W}} L_{\mathrm{W}}^{(n)}(\widehat{P}, P^\ast), \\
\text{(RKHS)} \quad & L^{(2)} = L_{\mathrm{fold}}(\widehat{P}, P^\ast) + \lambda_{\mathcal{H}} L_{\mathcal{H}}^{(n)}(\widehat{P}, P^\ast).
\end{aligned}
\]

Here \(n\) denotes **structural depth** (Cartesian with recipe). \(L_{\mathcal{H}}^{(n)}\) uses **sum-pooled** internal virtual objects on **both** prediction and native (§7–§8 of **Full differential pipeline**). \(L_{\mathrm{W}}^{(n)}\) compares \(D_a^{(n)}(\widehat{X})\) to \(D_a^{(n)}(X^\ast)\) over modules (same structural depth), with piecewise-smooth gradients through filtrations as in that section. **Do not** add \(\lambda_{\mathrm{land}} L_{\mathrm{land}}\); **persistence landscape** is **not** a stage.

#### 10–12. Scientific motivation and contribution (summary)
Proteins are modular and nested (neighborhoods → motifs → domains → global fold). Module comparison targets **relative topological organization of parts**; aggregation targets **interaction patterns among features**. A single global diagram difference \(D(\widehat{P})-D(P^\ast)\) is a valid virtual diagram but **does not** force the model to respect **how** topology is distributed and related inside the protein — the point of internal hierarchical drift.

**Concise contribution statement:** For static geometry, replace temporal drift by **internal hierarchical topological drift**: signed differences between persistence summaries of comparable substructures, lifted to higher structural levels via interval-of-interval aggregation, living in virtual persistence groups so the full RKHS pipeline applies.

**Mnemonic:** modules replace windows; internal comparison replaces temporal comparison; higher-level aggregation replaces naive first-order drift; virtual persistence still comes from formal differences; RKHS features are built on those signed higher-order objects.

### Full differential pipeline: competing stages × depth (authoritative derivation)

This subsection consolidates the **form backpropagation story** for **end-to-end fine-tuning** of a local **ColabFold / AF2-class** model with **topology losses that are not offline diagnostics**: they must contribute gradients to \(\omega\). **Do not conflate** the two literate sources:

- **Recursive depth** \(D^{(n)}\), strengthened metrics \(d_1^{(n)}\), recursive Wasserstein \(W_p^{(n)}\), virtual groups \(K^{(n)}(X)\): the **internal depth note** (\(D^{(n+1)}(X)=D(X^{(n)},A^{(n)})\), etc.).
- **Characters \(\chi_\theta\), heat measure, spectral symbol \(\lambda(\theta)\), translation-invariant RKHS kernel**: the **Virtual Persistence RKHS** paper (\(K(X,A)\), torus dual, \(k_t\), \(F_{\nu_t}\)).
- **Differentiability of persistence coordinates** away from combinatorial breakpoints: **topological-loss / differentiable PH** references (birth–death as functions of filtration values on fixed strata).

The **experiment grid** is **Stage × Depth**: Stage \(\in\{\)Baseline, Wasserstein, RKHS\(\}\); depth \(n\in\{0,1,2,\ldots\}\) is **not** another stage. The **Wasserstein** and **RKHS** branches are **competing trainings**, not summed in one run for primary comparison.

**Implementation note (repo today):** `src/topology/` provides **encodings** (e.g. grid + RFF) that are **surrogates** for finite computation. The **mathematical object** below uses **direct character evaluation on the virtual sum** where stated; documentation must keep **VPD** (the signed monoid object) distinct from any **encoding vector** used in code.

---

#### 1. Formal setup

Let \(\omega\) denote trainable ColabFold / AF2-class parameters and

\[
\widehat{X}(\omega) = (\hat{x}_1(\omega),\ldots,\hat{x}_N(\omega)) \in (\mathbb{R}^3)^N.
\]

Let \(X^\ast = (x_1^\ast,\ldots,x_N^\ast)\) be the **native** structure (constant in \(\omega\)).

Let \(\mathcal{M}(P)\) be the **CATH-induced** family of comparable internal modules for protein \(P\), and \(\mathcal{C}(P)\subseteq \mathcal{M}(P)\times\mathcal{M}(P)\) a **finite** set of ordered comparable pairs \((a,b)\).

For each module \(a\), let \(R_a:(\mathbb{R}^3)^N\to(\mathbb{R}^3)^{N_a}\) restrict to residues of \(a\). Write

\[
\widehat{X}_a(\omega) = R_a\,\widehat{X}(\omega), \qquad X^\ast_a = R_a X^\ast.
\]

For branch \(s\in\{\,\widehat{\ },\ast\,\}\), module \(a\), and depth \(n\), let \(D_a^{(n)}(X^s)\) be the depth-\(n\) persistence object from the **recursive depth note**. At \(n=1\) this is the ordinary diagram on the chosen filtration; higher depths use

\[
D^{(n+1)}(X) = D\bigl(X^{(n)}, A^{(n)}\bigr), \quad d^{(n+1)} = W_p^{(n+1)}, \quad K^{(n)}(X) = K\!\bigl(D^{(n)}(X)\bigr).
\]

On virtual objects, the metric used in that framework is typified by

\[
\rho^{(n)}(\Gamma-\Lambda,\;\Gamma'-\Lambda') = W_1^{(n)}(\Gamma+\Lambda',\;\Gamma'+\Lambda).
\]

**Composed pipeline (schematic):**

\[
\omega \;\longmapsto\; \widehat{X}(\omega) \;\longmapsto\; \text{module filtrations} \;\longmapsto\; D^{(n)} \;\longmapsto\;
\begin{cases}
L_W \quad \text{(Wasserstein stage)} \\
L_{\mathcal{H}} \quad \text{(RKHS stage)}
\end{cases}
\]

---

#### 2. Generic smooth strata and where derivatives exist

All **classical** (\(C^1\)) formulas below hold on an **open stratum** where the following **combinatorial data are fixed**:

1. CATH module decomposition and comparison family \(\mathcal{C}(P)\);
2. identity of critical simplices / recursive states realizing births and deaths;
3. support cardinalities of the relevant persistence / virtual objects;
4. **optimal transport matchings** realizing Wasserstein costs;
5. active branch of every \(\min\) in the strengthened metric \(d_1^{(n)}\).

When any of these **switch**, the map is generally **not** \(C^1\); the honest object is a **set-valued / Clarke generalized (sub)gradient**. **Piecewise-smooth** analysis is the rigorous setting.

---

#### 3. Coordinates \(\to\) filtration values (common front-end)

Under the frozen graph policy: **backbone** edges have filtration \(0\); **spatial** contact edge \(e=(i,j)\) uses Euclidean length

\[
r_{ij}(\omega) = \hat{x}_i(\omega)-\hat{x}_j(\omega), \qquad \ell_{ij}(\omega)=\|r_{ij}(\omega)\|_2.
\]

For \(\ell_{ij}>0\),

\[
\frac{\partial \ell_{ij}}{\partial \hat{x}_i} = \frac{r_{ij}}{\|r_{ij}\|_2}, \qquad \frac{\partial \ell_{ij}}{\partial \hat{x}_j} = -\frac{r_{ij}}{\|r_{ij}\|_2}.
\]

Backbone edges contribute **zero** gradient w.r.t. \(\ell\) (constant \(0\)). For any scalar loss \(L\) depending on filtration values,

\[
\frac{\partial L}{\partial \omega}
= \sum_{(i,j)\in E_{\mathrm{spatial}}} \frac{\partial L}{\partial \ell_{ij}} \left\langle \frac{r_{ij}}{\|r_{ij}\|_2},\, \frac{\partial \hat{x}_i}{\partial\omega} - \frac{\partial \hat{x}_j}{\partial\omega} \right\rangle .
\]

Thus, given \(\partial L/\partial \ell_{ij}\), the rest is standard autodiff through the network.

---

#### 4. Ordinary persistence coordinates vs. filtration (depth \(n=1\))

On a stratum, each off-diagonal point is \(z_\alpha(\omega)=(b_\alpha(\omega),d_\alpha(\omega))\). Births and deaths are **filtration values** of named critical simplices, so for edge filtration \(\ell_e\),

\[
\frac{\partial b_\alpha}{\partial \ell_e} =
\begin{cases} 1 & e = e_{\mathrm{birth}}(\alpha) \\ 0 & \text{otherwise} \end{cases},
\qquad
\frac{\partial d_\alpha}{\partial \ell_e} =
\begin{cases} 1 & e = e_{\mathrm{death}}(\alpha) \\ 0 & \text{otherwise} \end{cases}
\]

while critical identities stay fixed. Hence

\[
\frac{\partial z_\alpha}{\partial \omega}
= \sum_e \Bigl(\frac{\partial b_\alpha}{\partial \ell_e},\, \frac{\partial d_\alpha}{\partial \ell_e}\Bigr)\frac{\partial \ell_e}{\partial \omega}.
\]

---

#### 5. Recursive depth: product metric, strengthened metric, \(W_p^{(n+1)}\)

Write \(X^{(n)} = D^{(n)}(X)\times D^{(n)}(X)\) with product metric

\[
d_{\mathrm{prod}}^{(n)}\bigl((\alpha_1,\alpha_2),(\beta_1,\beta_2)\bigr)
= \bigl\|\bigl(d^{(n)}(\alpha_1,\beta_1),\, d^{(n)}(\alpha_2,\beta_2)\bigr)\bigr\|_p,
\]

**strengthened metric**

\[
d_1^{(n)}(u,v) = \min\Bigl( d_{\mathrm{prod}}^{(n)}(u,v),\; d_{\mathrm{prod}}^{(n)}(u,A^{(n)}) + d_{\mathrm{prod}}^{(n)}(v,A^{(n)}) \Bigr),
\]

and recursive Wasserstein

\[
W_p^{(n+1)}(\Gamma,\Lambda) = \inf_{\sigma} \bigl\| \bigl(d_1^{(n)}(u_i,v_{\sigma(i)})\bigr)_i \bigr\|_p
\]

(in standard pairing notation).

**5.1 Product metric.** If \(\delta_1 = d^{(n)}(\alpha_1,\beta_1)\), \(\delta_2 = d^{(n)}(\alpha_2,\beta_2)\) and \(d_{\mathrm{prod}}^{(n)} = (\delta_1^p+\delta_2^p)^{1/p}\) with \(\delta_1,\delta_2>0\),

\[
\frac{\partial d_{\mathrm{prod}}^{(n)}}{\partial \delta_1}
= \frac{\delta_1^{p-1}}{(\delta_1^p+\delta_2^p)^{(p-1)/p}},
\qquad
\frac{\partial d_{\mathrm{prod}}^{(n)}}{\partial \delta_2}
= \frac{\delta_2^{p-1}}{(\delta_1^p+\delta_2^p)^{(p-1)/p}},
\]

so \(\nabla_\omega d_{\mathrm{prod}}^{(n)}\) splits into \(\nabla_\omega d^{(n)}(\alpha_1,\beta_1)\) and \(\nabla_\omega d^{(n)}(\alpha_2,\beta_2)\) with those coefficients.

**5.2 Strengthened metric.** Let \(u_1=d_{\mathrm{prod}}^{(n)}(u,v)\), \(u_2=d_{\mathrm{prod}}^{(n)}(u,A^{(n)})+d_{\mathrm{prod}}^{(n)}(v,A^{(n)})\), so \(d_1^{(n)}(u,v)=\min(u_1,u_2)\). On a stratum: if \(u_1<u_2\), \(\nabla d_1^{(n)}=\nabla u_1\); if \(u_2<u_1\), \(\nabla d_1^{(n)}=\nabla u_2\); if \(u_1=u_2\), use **\(\operatorname{co}\{\nabla u_1,\nabla u_2\}\)** (Clarke generalized gradient).

**5.3 Recursive \(W_p^{(n+1)}\).** With unique optimal matching, write \(c_i = d_1^{(n)}(u_i,v_i)\) and \(W_p^{(n+1)} = (\sum_i c_i^p)^{1/p}\). Then

\[
\frac{\partial W_p^{(n+1)}}{\partial c_i}
= \frac{c_i^{p-1}}{(W_p^{(n+1)})^{p-1}},
\qquad
\nabla_\omega W_p^{(n+1)} = \sum_i \frac{c_i^{p-1}}{(W_p^{(n+1)})^{p-1}}\, \nabla_\omega d_1^{(n)}(u_i,v_i).
\]

This **recurses** until depth \(1\), then Section 3–4, then \(\partial\omega\) through \(\widehat{X}\).

---

#### 6. Stage 1 — Wasserstein branch

**Semantic:** compare **predicted vs native** topology objects at the **same** structural resolution (module / lifted depth).

Example depth-\(n\) loss (summed over the chosen module family \(\mathcal{M}(P)\) or other fixed aggregation):

\[
L_W^{(n)}(\omega) = \sum_{a\in\mathcal{M}(P)} W_p^{(n)}\!\Bigl( D_a^{(n)}(\widehat{X}(\omega)),\, D_a^{(n)}(X^\ast) \Bigr).
\]

**Stage-1 total objective** (one training run; \(\lambda_{W,n}\ge 0\)):

\[
L_{\mathrm{stage},1}(\omega) = L_{\mathrm{fold}}(\omega) + \sum_{n\in\mathcal{N}_W} \lambda_{W,n}\, L_W^{(n)}(\omega).
\]

**6.1 Depth \(n=1\) transport.** Ground cost between \(z=(b,d)\) and \(z^\ast=(b^\ast,d^\ast)\) in \(\ell_\infty\) diagram geometry:

\[
c(z,z^\ast)=\|z-z^\ast\|_\infty = \max\{|b-b^\ast|,\,|d-d^\ast|\}.
\]

Off ties: if \(|b-b^\ast|>|d-d^\ast|\), \(\partial c/\partial b=\operatorname{sgn}(b-b^\ast)\), \(\partial c/\partial d=0\); swapped if \(|d-d^\ast|>|b-b^\ast|\); at equality, **subgradient** of the max. **Diagonal** penalty often uses \(c_\Delta(b,d)= (d-b)/2\), so \(\partial c_\Delta/\partial b=-\tfrac12\), \(\partial c_\Delta/\partial d=+\tfrac12\). Then \(\nabla_\omega c = (\partial c/\partial b)\nabla_\omega b + (\partial c/\partial d)\nabla_\omega d\) with Section 4.

**6.2 Higher depth:** costs are \(d_1^{(n-1)}\) terms inside \(W_p^{(n)}\); differentiate with Section 5 recursively. Schematically, with module \(a\) and optimal pairs \((u_{a,i},v_{a,i})\) on a stratum,

\[
\frac{\partial L_W^{(n)}}{\partial \omega}
= \sum_{a} \sum_{i} \frac{c_{a,i}^{p-1}}{\bigl(W_p^{(n)}\bigr)^{p-1}}\,
\frac{\partial}{\partial \omega} d_1^{(n-1)}\bigl(u_{a,i},v_{a,i}\bigr),
\]

unpacking \(d_1^{(n-1)}\) by Section 5 until depth \(1\), then Sections 2–4.

---

#### 7. Stage 2 — RKHS semimetric branch

**Semantic:** **RKHS distance on virtual objects** built from the **internal VPD family** at depth \(n\), **separate** from the Wasserstein stage (no adding both in one primary recipe).

For each \((a,b)\in\mathcal{C}(P)\) and depth \(n\), define internal virtual differences (sign is part of the **virtual persistence** formalism)

\[
G_{a,b}^{(n)}(\widehat{X}) = D_a^{(n)}(\widehat{X}) - D_b^{(n)}(\widehat{X}) \in K^{(n)}(X), \qquad
G_{a,b}^{(n)}(X^\ast) = D_a^{(n)}(X^\ast) - D_b^{(n)}(X^\ast).
\]

**Pooling (frozen for RKHS in this design): sum pooling** over comparable pairs:

\[
\Gamma_{\widehat{X}}^{(n)}(\omega) = \sum_{(a,b)\in\mathcal{C}(P)} G_{a,b}^{(n)}(\widehat{X}(\omega)), \qquad
\Gamma_{X^\ast}^{(n)} = \sum_{(a,b)\in\mathcal{C}(P)} G_{a,b}^{(n)}(X^\ast).
\]

Let \(\Phi_t\) map a **virtual group element** into the RKHS at heat time \(t\) induced by the VPD paper (\(k_t\), characters, spectral symbol). The depth-\(n\) RKHS loss is

\[
L_{\mathcal{H}}^{(n)}(\omega) = \left\|\Phi_t\!\bigl(\Gamma_{\widehat{X}}^{(n)}(\omega)\bigr) - \Phi_t\!\bigl(\Gamma_{X^\ast}^{(n)}\bigr)\right\|_{\mathcal{H}_t}^2 .
\]

By the **reproducing-kernel identity**,

\[
L_{\mathcal{H}}^{(n)}(\omega)
= k_t\!\bigl(\Gamma_{\widehat{X}}^{(n)},\Gamma_{\widehat{X}}^{(n)}\bigr)
+ k_t\!\bigl(\Gamma_{X^\ast}^{(n)},\Gamma_{X^\ast}^{(n)}\bigr)
- 2\,\Re\, k_t\!\bigl(\Gamma_{\widehat{X}}^{(n)},\Gamma_{X^\ast}^{(n)}\bigr).
\]

With translation invariance \(k_t(\alpha,\beta)=k_t(\alpha-\beta,0)\) and \(\Delta\Gamma^{(n)}(\omega)=\Gamma_{\widehat{X}}^{(n)}(\omega)-\Gamma_{X^\ast}^{(n)}\),

\[
L_{\mathcal{H}}^{(n)}(\omega) = 2\Bigl( k_t(0,0) - \Re\, F_{\nu_t}\bigl(\Delta\Gamma^{(n)}(\omega)\bigr) \Bigr),
\qquad
F_{\nu_t}(\gamma) = \int \chi_\theta(\gamma)\, e^{-t\lambda(\theta)} \, d\mu(\theta)
\]

(in the paper’s normalizations).

**Stage-2 total objective:**

\[
L_{\mathrm{stage},2}(\omega) = L_{\mathrm{fold}}(\omega) + \sum_{n\in\mathcal{N}_{\mathcal{H}}} \lambda_{\mathcal{H},n}\, L_{\mathcal{H}}^{(n)}(\omega), \quad \lambda_{\mathcal{H},n}\ge 0.
\]

---

#### 8. RKHS branch — exact derivative skeleton

Write the kernel with explicit \(\omega\)-dependence of the spectral side and torus dimension on the current stratum as

\[
k_t^{(\omega)}(\alpha,\beta)
= \int_{T^{N_\omega}} \chi_\theta(\alpha-\beta)\, e^{-t\lambda_\omega(\theta)} \, d\mu(\theta).
\]

Define

\[
\Psi(\theta,\omega) = \chi_\theta\bigl(\Delta\Gamma^{(n)}(\omega)\bigr)\, e^{-t\lambda_\omega(\theta)}.
\]

Then \(L_{\mathcal{H}}^{(n)}(\omega) = 2\bigl( k_t(0,0) - \Re \int \Psi(\theta,\omega)\, d\mu(\theta) \bigr)\). On a stratum where differentiation under the integral is valid,

\[
\frac{\partial L_{\mathcal{H}}^{(n)}}{\partial \omega}
= -2\,\Re \int \frac{\partial \Psi(\theta,\omega)}{\partial \omega}\, d\mu(\theta).
\]

**Product rule:**

\[
\frac{\partial \Psi}{\partial \omega}
= \frac{\partial \chi_\theta(\Delta\Gamma^{(n)}(\omega))}{\partial \omega}\, e^{-t\lambda_\omega(\theta)}
- t\,\chi_\theta(\Delta\Gamma^{(n)}(\omega))\, e^{-t\lambda_\omega(\theta)}\, \frac{\partial \lambda_\omega(\theta)}{\partial \omega}.
\]

Hence the **boxed branch formula**

\[
\boxed{
\frac{\partial L_{\mathcal{H}}^{(n)}}{\partial \omega}
= -2\,\Re \int e^{-t\lambda_\omega(\theta)}
\left[
\frac{\partial \chi_\theta(\Delta\Gamma^{(n)}(\omega))}{\partial \omega}
- t\,\chi_\theta(\Delta\Gamma^{(n)}(\omega))\, \frac{\partial \lambda_\omega(\theta)}{\partial \omega}
\right] d\mu(\theta)
}
.
\]

---

#### 9. Character term — **direct evaluation on the virtual object** (no hard histogram as the formal gradient story)

Fix a stratum where \(\Delta\Gamma^{(n)}(\omega)=\sum_{j=1}^M m_j\,\xi_j(\omega)\) with **fixed integers** \(m_j\) and smooth atoms \(\xi_j(\omega)\). A **direct character map** uses phase functions \(\varphi_\theta\) associated to the dual torus:

\[
\chi_\theta\bigl(\Delta\Gamma^{(n)}(\omega)\bigr)
= \exp\!\Bigl( i \sum_{j=1}^M m_j\, \varphi_\theta\bigl(\xi_j(\omega)\bigr) \Bigr).
\]

Then

\[
\boxed{
\frac{\partial \chi_\theta\bigl(\Delta\Gamma^{(n)}(\omega)\bigr)}{\partial \omega}
= i\,\chi_\theta\bigl(\Delta\Gamma^{(n)}(\omega)\bigr)
\sum_{j=1}^M m_j\,
\nabla_\xi \varphi_\theta\bigl(\xi_j(\omega)\bigr) \cdot \frac{\partial \xi_j(\omega)}{\partial \omega}
}
.
\]

Each \(\xi_j\) is a depth-\(n\) atom: at \(n=1\), \(\xi_j=(b_j,d_j)\) and \(\partial\xi_j/\partial\omega\) comes from Section 4; at higher \(n\), \(\xi_j\) lives in the recursive interval-of-intervals / lifted picture and \(\partial\xi_j/\partial\omega\) follows Section 5 **composed backward** to depth \(1\). **This is where the depth calculus and the RKHS calculus meet.**

---

#### 10. Spectral symbol \(\lambda_\omega(\theta)\)

From the paper’s graph / Laplacian picture (schematic),

\[
\lambda_\omega(\theta) = \sum_{\{u,v\}\in E_\omega} w_{uv}(\omega)\Bigl( 1 - \cos \Delta_{uv}(\theta,\omega) \Bigr),
\]

\[
\Delta_{uv}(\theta,\omega) = \operatorname{dist}_{\mathbb{R}/2\pi\mathbb{Z}}\Bigl( \varphi_\theta(u(\omega)),\, \varphi_\theta(v(\omega)) \Bigr).
\]

Differentiate:

\[
\boxed{
\frac{\partial \lambda_\omega(\theta)}{\partial \omega}
= \sum_{\{u,v\}} \bigl(1-\cos\Delta_{uv}\bigr)\, \frac{\partial w_{uv}}{\partial \omega}
+ \sum_{\{u,v\}} w_{uv}\,\sin\Delta_{uv}\, \frac{\partial \Delta_{uv}}{\partial \omega}
}
.
\]

**10.1 Weights.** If \(w_{uv}=d(u,v)\) in the paper’s convention, then \(\partial w_{uv}/\partial\omega\) is the derivative of that **metric on virtual configuration / lifted vertices**; at depth \(1\) it reduces to plane geometry on \((b,d)\); at depth \(>1\) it invokes derivatives of **recursive \(W_p\)** (Section 5).

**10.2 Circle gaps.** On a stratum with a fixed geodesic representative of the wrap,

\[
\frac{\partial \Delta_{uv}}{\partial \omega}
= \operatorname{sgn}_{\mathrm{arc}}\Bigl(\varphi_\theta(u)-\varphi_\theta(v)\Bigr)\,
\left(
\nabla \varphi_\theta(u)\cdot \frac{\partial u}{\partial\omega}
- \nabla \varphi_\theta(v)\cdot \frac{\partial v}{\partial\omega}
\right),
\]

with the usual **subgradient** at branch-cut ties.

---

#### 11. Final RKHS gradient (substitute §9–§10 into §8)

\[
\boxed{
\begin{aligned}
\frac{\partial L_{\mathcal{H}}^{(n)}}{\partial \omega}
&= -2\,\Re \int e^{-t\lambda_\omega(\theta)}
\Biggl[
i\,\chi_\theta\bigl(\Delta\Gamma^{(n)}(\omega)\bigr)
\sum_{j=1}^M m_j\,
\nabla_{\xi}\varphi_\theta\bigl(\xi_j(\omega)\bigr) \cdot \frac{\partial \xi_j(\omega)}{\partial \omega} \\[0.6em]
&\qquad\qquad\qquad
-\; t\,\chi_\theta\bigl(\Delta\Gamma^{(n)}(\omega)\bigr)\,
\frac{\partial \lambda_\omega(\theta)}{\partial \omega}
\Biggr] d\mu(\theta),
\end{aligned}
}
\]

with \(\partial\lambda_\omega/\partial\omega\) from §10; every \(\partial\xi_j\), \(\partial u\), \(\partial v\), \(\partial w\) **chains through recursive depth** then **filtration lengths** then **\(\widehat{X}(\omega)\)**.

---

#### 12. Combined training on the two **competing** branches

\[
\frac{\partial L_{\mathrm{stage},1}}{\partial \omega}
= \frac{\partial L_{\mathrm{fold}}}{\partial \omega}
+ \sum_n \lambda_{W,n}\, \frac{\partial L_W^{(n)}}{\partial \omega},
\qquad
\frac{\partial L_{\mathrm{stage},2}}{\partial \omega}
= \frac{\partial L_{\mathrm{fold}}}{\partial \omega}
+ \sum_n \lambda_{\mathcal{H},n}\, \frac{\partial L_{\mathcal{H}}^{(n)}}{\partial \omega}.
\]

If \(\lambda\)’s are learnable with a positive reparameterization \(\lambda=\mathrm{softplus}(\alpha)\),

\[
\frac{\partial L}{\partial \alpha}
= \frac{\partial L}{\partial \lambda}\, \sigma(\alpha)
\]

(\(\sigma\) logistic), and \(\partial L_{\mathrm{stage},1}/\partial\lambda_{W,n} = L_W^{(n)}\), \(\partial L_{\mathrm{stage},2}/\partial\lambda_{\mathcal{H},n} = L_{\mathcal{H}}^{(n)}\) when those \(\lambda\)’s are free scalar knobs.

---

#### 13. Smooth vs. piecewise smooth (rigor checklist)

**Classically differentiable** on each stratum with **fixed**: critical / recursive identities, transport matchings, virtual support, \(\min\) branches, **unique** OT plans where used, circle **geodesic branches**.

**Nonsmooth boundaries** where those switch: \(\max/\min\), pairing changes, **non-unique transport**, persistence cancellations, circle tie events — use **subgradients / Clarke differentials**, not a single Jacobian.

---

#### 14. Correction record (terminology)

- **Depth** \(D^{(n)}\), \(W_p^{(n)}\), \(K^{(n)}\): **recursive VPD depth note**, not the RKHS paper alone.
- **Characters, heat kernel, \(\lambda(\theta)\), RKHS semimetric**: **VPD RKHS** paper.
- **Direct character on \(\Delta\Gamma\)**: the **exact Fourier–Stieltjes / character** derivative here; **histogram / coarse binning** is **not** the formal differential story for this branch (optional implementation approximation only, and must not be misnamed as “the VPD” in documentation).

<!-- Optional LaTeX export: the same content can be lifted to an Overleaf subsection with Assumption / Proposition / Corollary environments. -->

---

### Planned model ladder (ColabFold / AF2-class baseline; implement incrementally)
- Data unit: single CATH domain from PDB mmCIF; splits by superfamily (S35 helpers optional); CASP-style external check later.
- **Recipe A — Baseline:** \(L = L_{\mathrm{AF2}}\) only (smoke today: **inference** via `colabfold_batch`; weights on first run).
- **Recipe B — Wasserstein:** \(L = L_{\mathrm{AF2}} + \lambda_{\mathrm{W}} L_{\mathrm{W}}\) (**Wasserstein / diagram** loss at chosen depth \(n\); **not** persistence landscape).
- **Recipe C — RKHS:** \(L = L_{\mathrm{AF2}} + \lambda_{\mathcal{H}} L_{\mathcal{H}}\) — RKHS semimetric on **sum-pooled** internal VPDs (formal: **characters + heat kernel** on \(\Delta\Gamma^{(n)}\); see README **Full differential pipeline**). **`src/topology/`** uses **grid + RFF** as a **computable surrogate** for the reference heat-kernel/RFF stack (**DEF-K01–K03**, **DEF-L01–L02**), not as the mathematical definition of the VPD.
- **Depth \(n\)** runs as a **grid** over recipes B and C, not a third “stage name.”
- **Primary metrics:** **TM-score**, **GDT-TS**. **Secondary:** CA RMSD, pLDDT. **Training losses** \(L_{\mathrm{W}}, L_{\mathcal{H}}\) reported as **optimization terms**, not headline benchmarks. Smoke PNG: **Level** (diagnostic H0/H1/combo — **not** recursive depth \(n\)), **Stage**, **TM-score**, **pLDDT** (`smoke_table_render.py`; TM-align via `install_tm_align_linux.sh` when available). Details: `src/protein/smoke_nine_experiments.py`.

### Frozen dataset selection spec (`src/protein/dataset_policy.py`)
- Spec id: `protein_domain_cath_rcsb_v1` (`FROZEN_SELECTION_SPEC_ID`).
- RCSB entry filters: `structure_determination_methodology == experimental`, (X-ray and `resolution_combined <= 3.0 A`) OR (cryo-EM and `resolution_combined <= 4.0 A`), and `deposited_polymer_entity_instance_count == 1` for the Search seed list (strict single-instance universe).

### PDB extraction contract (metadata-first, NeurIPS-style auditing)
- **Goal:** each PDB entry is a benchmark row with cohort metadata (modality, resolution band, ligand state, membrane flags, validation summaries) before any coordinate training. The working coordinate object is the **deposited asymmetric-unit mmCIF** from `files.rcsb.org/download/<PDB>.cif`, not a biological assembly reconstruction. Biological assemblies are out of scope for the core pass.
- **Manifest schema:** `FROZEN_PDB_MANIFEST_VERSION` and column order in `src/protein/pdb_manifest_schema.py`. Rows are produced from the **RCSB Data API** (`https://data.rcsb.org/rest/v1/core/entry/{id}` plus the first protein `polymer_entity`). `validation_quality_fields` stores compact JSON derived from `pdbx_vrpt_summary*` blocks (for example EM `qscore`, X-ray clashscore geometry summaries). `pdb_stratum` assigns `core_xray`, `cryo_em_robustness`, or `stress` for structured heterogeneity (high-resolution X-ray core; cryo-EM robustness; stress = multimers, membrane or ligand/cofactor signals, long chains, or moderate X-ray beyond the core band). `entry_exclusion_code` is a semicolon-separated cascade (for example `multiple_polymer_instances`, `length_outside_80_800`, non-`REL` status).
- **Pass 1 (metadata only):** `python -u scripts/protein/build_pdb_metadata_manifest.py --entry-ids PATH.csv` or `--from-search --max-entries N` writes `results/protein/output/pdb_metadata_manifest.csv` with no mmCIF bytes.
- **Pass 2 (coordinates + validation PDF):** `python -u scripts/protein/download_pdb_entry_artifacts.py --manifest IN.csv --out-manifest OUT.csv` writes mmCIF under `data/protein/mmcif/` and validation PDFs under `data/protein/validation/`. mmCIF uses `files.rcsb.org/download` with divided `.cif.gz` fallback. Validation PDFs try **PDBe** first (`www.ebi.ac.uk/pdbe/entry-files/{pdb}_validation.pdf`), then RCSB and PDBj paths, because some RCSB validation URLs return HTTP 403 to scripted clients. SHA256 columns and `download_status` record what succeeded.
- **PDB-only analytic unit:** one deposited protein polymer entity with a representative chain id (first sorted `pdbx_strand_id` token when the ASU lists multiple chain copies). Chain-level parsing outcomes stay in `chain_length_parsed`, `parse_status`, `graph_status`, and `topology_status` (filled by later enrichment scripts).
- CATH inputs: CATH List File (CLF) 2.0 domain lists such as `cath-domain-list.txt` (or `cath-domain-list-S35.txt` for redundancy control).
- Domain length: 80 to 800 residues (CATH column 11).
- CATH resolution column: exclude `>= 999.0` (NMR / obsolete flags in CATH) and require `<= 4.0 A` after exclusions.
- Phase-1 domain chopping: only whole-chain CATH domains (`domain_id` ends in `00`). Chopped domains require `cath-domain-boundaries.txt` parsing (later milestone).
- Chain placeholder: rows with CATH chain character `0` are excluded until a dedicated loader exists.
- Split key for non-leaky evaluation: `superfamily` (`C.A.T.H` node from CLF); optional secondary clustering via `s35_cluster` when using S35 lists.
- Smoke sampling: `scripts/protein/sample_smoke_domains.py` picks `target_superfamilies` and `per_superfamily` with RNG seed `14`.

### Topology object (graph-first clique filtration; preprocessing / loss only)
- **Frozen primary graph** (`TOPOLOGY_GRAPH_POLICY_ID` in `src/protein/dataset_policy.py`): residue nodes at **C_beta** when present, **virtual C_beta** from `(N,CA,C)` when CB is missing, **C_alpha for glycine**. Edges: mandatory backbone `(i,i+1)` at **filtration 0**, plus spatial contacts for `|i-j|>1` with **Euclidean distance `<= 8 A`** and filtration value **`d_ij`**. Clique complex and PH via `src/protein/clique_persistence.py`. Ablate with `--graph-mode ca_legacy` (C_alpha-only, all edges use distance as in the older prototype).
- **Separation from ColabFold/AlphaFold2:** internal pair representations in the folding model are **orthogonal** to this definition. Here the graph exists solely to define a **scalar-edge filtration** and clique complex for PH on **coordinates** (native in the cache; predicted when computing \(L_{\mathrm{top}}\)). Do not describe the protein system as “TGN-like” or as using this graph as the main **predictive** encoder.
- Cache: `python -u scripts/protein/cache_topology_features_mmcif.py --mmcif-path <file> [--chain-id C]` writes `data/processed/protein/*.npz` (embeds `mmcif_sha256`, policy id, coords, edges, persistence). Rebuild with `--force` after changing mmCIF or policy.

### Repository locations
- Policy constants: `src/protein/dataset_policy.py`.
- CATH CLF streaming parser: `src/protein/cath_clf.py`.
- RCSB pagination client: `src/protein/rcsb_client.py`.
- PDB manifest + Data API row builder: `src/protein/rcsb_data_api.py`, `src/protein/pdb_strata.py`, `src/protein/validation_urls.py`.
- Eligibility filter: `src/protein/eligible_domains.py`.
- mmCIF CA loader: `src/protein/mmcif_io.py` (optional `--chain-id`).
- Residue representatives and graph: `src/protein/residue_points.py`, `src/protein/residue_graph.py`, `src/protein/clique_persistence.py`, `src/protein/topology_cache.py`.

### Protein scripts (metadata-first order)
1. `python -u scripts/protein/fetch_rcsb_entry_ids.py --out results/protein/output/rcsb_entry_ids.csv` (optional `--max-ids N`; optional `--page-size`).
2. `python -u scripts/protein/build_pdb_metadata_manifest.py --entry-ids-csv results/protein/output/rcsb_entry_ids.csv --out results/protein/output/pdb_metadata_manifest.csv` **or** `--from-search --max-entries N --out ...` (full NeurIPS-style PDB row before CATH).
3. `python -u scripts/protein/download_pdb_entry_artifacts.py --manifest results/protein/output/pdb_metadata_manifest.csv --out-manifest results/protein/output/pdb_metadata_manifest_downloaded.csv` (mmCIF + validation PDF pass).
4. Download CATH classification files locally (for example `cath-domain-list.txt`).
5. `python -u scripts/protein/build_eligible_domain_manifest.py --rcsb-ids results/protein/output/rcsb_entry_ids.csv --cath-domain-list <PATH_TO_cath-domain-list.txt> --out results/protein/output/eligible_domains.csv`
6. `python -u scripts/protein/sample_smoke_domains.py --manifest results/protein/output/eligible_domains.csv --out results/protein/output/smoke_domains.csv`
7. `python -u scripts/protein/download_manifest_mmcif.py --manifest results/protein/output/smoke_domains.csv --out-dir data/protein/mmcif --skip-existing` (CATH-driven subset downloads; prefer aligning with the PDB manifest when both exist).
8. `python -u scripts/protein/compute_clique_persistence_mmcif.py --mmcif-path data/protein/mmcif/<PDB>.cif --chain-id <CHAIN>`
- Structure figures (py3Dmol + Playwright): after `python -m playwright install chromium`, run `python -u scripts/protein/render_structure_cartoon_surface.py --mmcif-dir data/protein/mmcif` to write `results/protein/figures/structure_views/<PDB>_cartoon.png` (3Dmol secondary-structure cartoon, rainbow) and `<PDB>_surface.png` (VDW surface on the selected chain).
- Corpus utilities: `scripts/protein/list_mmcif_corpus.py`, `scripts/protein/smoke_mmcif_parse.py --mmcif-path <file> [--chain-id <C>]`.

### Smoke report interface (Layers 1–3 inference + training regime rows)
- **Upstream:** [ColabFold](https://github.com/sokrypton/ColabFold) (batch CLI, public MSA server for small jobs). **Local install:** [LocalColabFold](https://github.com/YoshitakaMo/localcolabfold) (pixi-based; **not** the same Python env as the project `.venv-wsl-gpu`).
- **Windows (native):** `python -u scripts/protein/run_smoke_pipeline.py` runs Layers 1–2 and **aborts** before ColabFold unless you pass `--colabfold-dry-run` (FASTA only) or `--no-colabfold-smoke`. Real `colabfold_batch` smoke **must** run inside **WSL2** with GPU (see below).
- **WSL2 + NVIDIA GPU (required for real M1 predict):**
  1. **Filesystem:** On WSL, `$HOME` is **ext4**; `C:\` is **`/mnt/c`** (**9p**, slow for huge installs). `wsl_run_smoke` **refuses `/mnt/c/...`**. Copy a **minimal** tree: `bash scripts/protein/wsl_sync_minimal_to_home.sh`. Inspect: `bash scripts/protein/wsl_probe_env.sh`.
  2. `cd "$HOME/2_Protein_Folding"` (or `WSL_HOME_REPO`), then `bash scripts/protein/wsl_setup_protein_venv.sh` (project venv: torch **cu124**, topology + figures). If torch stack was wrong: `bash scripts/protein/wsl_repair_torch_cu124.sh` (ColabFold path; add `ENABLE_LEGACY_OPENFOLD3_REPAIR=1` **only** if repairing the embargoed legacy venv).
  3. `bash scripts/protein/wsl_install_localcolabfold.sh` — installs **ColabFold** under `$HOME/localcolabfold` (or `LOCALCOLABFOLD_ROOT`). AlphaFold weights download on **first** `colabfold_batch` run.
  4. `bash scripts/protein/wsl_run_smoke.sh --skip-structure-figures` (prepends LocalColabFold `bin` to `PATH` when present).
  - **GPU / CUDA (LocalColabFold upstream):** GPU JAX builds expect a **CUDA toolkit** compatible with their stack (upstream docs: **CUDA ≥ 12.1**, cudnn; **CUDA 12.4** often recommended). Check the **compiler** with `nvcc --version` inside WSL — the **driver** reported by `nvidia-smi` is related but not the same thing. If you see `CUDA_ERROR_ILLEGAL_ADDRESS` or similar, fix toolkit/driver alignment before blaming the smoke scripts.
  - **MSA / databases:** Default small-batch flow uses the **public ColabFold MSA server** (no local **~940 GB** DB). Offline / fully local search is optional and separate ([`setup_databases.sh` in ColabFold](https://github.com/sokrypton/ColabFold/blob/main/setup_databases.sh)).
  - **WSL symlink issues:** If `pixi install` / setup fails on symlinks under a Windows-backed tree, install only under **`$HOME` (ext4)** (what `wsl_install_localcolabfold.sh` enforces), or per [LocalColabFold](https://github.com/YoshitakaMo/localcolabfold) enable **case-sensitive** NTFS for that folder from **Windows PowerShell**: `fsutil file SetCaseSensitiveInfo <path> enable` (not from WSL).
  - Layer 3 smoke checks **nvidia-smi** (GPU present) before `colabfold_batch` (`src/protein/cuda_guard.py`); **no CPU fallback** for structure inference in this smoke.
  - **Low VRAM / laptop GPUs:** pass tuning flags via `--colabfold-extra-args` (forwarded to `colabfold_batch`), e.g. smaller MSA caps or fewer recycles — see `colabfold_batch --help` and upstream **Flags** (e.g. `--max-msa`, `--num-recycle`).
  - GPU metadata: `results/protein/output/colabfold_smoke/smoke_cuda_validation.json`, status `layer_c_colabfold_smoke.csv`.
- Other flags: `--corpus-dir`, `--force-topology`, `--skip-structure-figures`, `--no-colabfold-smoke`, `--colabfold-dry-run`, `--colabfold-max-structures N`, `--colabfold-extra-args`, `--colabfold-allow-missing-cli`.
- **Layers 1–2 (data pipeline):** structure tables, topology NPZ stats, bar charts (transparent PNG), **3D persistence assets** per PDB (`figures/smoke/assets/<PDB>_pd_3d.png`, same engine as `scripts/figures/persistence_diagram_3d_engine.py`), **residue PCA graph** (`figures/smoke/assets/smoke_residue_graph_pca.png`), and a **nine-run metrics table** CSV + PNG (`smoke_nine_run_metrics.csv`, `figures/smoke/assets/table_smoke_nine_run.png`, epi-style grid — no heatmap). See `tables/smoke/smoke_artifacts_manifest.csv`.
- **Layer 3 — ColabFold:** `scripts/protein/run_colabfold_predict_smoke.py` writes FASTA, runs `colabfold_batch`. **Not wired:** anything under `scripts/protein/legacy/` (see `legacy/README.md`).
- **Layer 3 — M2–M4:** `smoke_pipeline_status.csv` lists training regimes only.
- **Outputs:** `results/protein/tables/smoke/*`, `results/protein/figures/smoke/*` (including `structure_views/` when py3Dmol runs), `results/protein/output/smoke_colabfold_probe.json`, `results/protein/output/colabfold_smoke/layer_c_colabfold_smoke.csv`. **`smoke_parse_errors.csv`:** mmCIF files that **failed frozen representative rules** (e.g. missing CA / backbone for virtual Cβ) — the files are still in `data/protein/mmcif/`; they are **excluded from Layer A/B rows**, not “missing from the dataset.”

### Shared reuse (math only; not architecture)
- `src/topology/` holds persistence summaries, virtual-diagram tooling, and RKHS / alignment-style losses developed for **epidemiology**. For proteins, reuse is **limited to auxiliary objectives** (and shared notation), not TGN checkpoints or temporal encoders. Clique persistence for proteins lives in `src/protein/clique_persistence.py` with filtrations matched to the residue graph policy.

---

## Protein folding: full experimental pipeline (reconstruction guide)

**Status:** Smoke tables use stage **`Wasserstein`** for the \(W_2\) rows (**DEF-S02**). Semantics: **Design freeze** above.

This section ties **motivation**, **mathematics**, **statistics**, and **code paths** so a reader can reproduce the protein track and see what is implemented today versus what the formal note targets for full training.

### 1. Scientific goal and estimand

- **Question:** whether **competing fine-tuning recipes** that add **either** a **Wasserstein diagram loss** **or** an **RKHS loss on virtual persistence diagrams** to the standard **ColabFold / AF2-class** objective improve **held-out** structure quality under **hard splits** (e.g. CATH homologous superfamily), **without** feeding topology vectors into the folding trunk.
- **Primary research metrics** (real benchmarks — not smoke): **GDT-TS**; co-primary **TM-score** and **lDDT** (often from external tools such as TM-align or OpenStructure). These differ from epidemiology’s **RMSE / Brier / ECE**, which score probabilistic calibration of a scalar epidemic outcome.
- **Smoke / dev metrics:** **per-residue CA RMSD after Kabsch** (global, in `smoke_per_structure_metrics.csv`), **mean CA pLDDT** from prediction PDB B-factors, plus the **nine-cell module report** (see §9). Smoke is for **pipeline validation**, not publication benchmarks.

### 2. Epidemiology analogy (what transfers and what does not)

- **Transfers:** persistence diagrams, **virtual / signed** diagram objects, **heatmap-kernel** construction via **random Fourier features** (`HeatRandomFeatures` in `src/topology/kernels.py`) and **TopologicalRKHSLoss** in `src/topology/loss.py`; idea of **multi-stage** training (baseline → add PH → add RKHS).
- **Does not transfer:** TGN encoders, temporal windows, epidemic simulators, `src/models/` checkpoints, and any **concatenation** of topology into a sequence/graph backbone for **folding**. Proteins use **static** 3D coordinates; “drift” is **internal** (module-vs-module), not time-series (see formal theory above).

### 3. Coordinate representation and graph policy (frozen)

**Implementation:** `TOPOLOGY_GRAPH_POLICY_ID`, `CONTACT_GRAPH_RADIUS_MAX_A = 8` Å, `src/protein/dataset_policy.py`, `residue_points.py`, `residue_graph.py`.

- **Nodes:** one per residue, at **Cβ** when present; **virtual Cβ** from (N, CA, C) when Cβ is missing; **Cα for glycine** (`cb_topology`). Optional `ca_legacy`: Cα only, older edge rule.
- **Edges:** (i) **Backbone** consecutive pairs \((i,i+1)\) with **filtration value 0** (appear at the start of the filtration). (ii) **Spatial contacts:** \(|i-j|>1\) and Euclidean distance \(d_{ij} \le 8\) Å; edge filtration = **\(d_{ij}\)** (in Å).

**Motivation:** this yields a **metric/threshold graph** whose clique complex reflects **geometric proximity** at multiple scales; it is **orthogonal** to AF2’s learned pair stack in the sense that it is a **fixed geometric construction** on coordinates, not a learned pairwise tensor from the model.

**Statistical note:** the 8 Å cap is a **policy constant** (literature often uses ~7–8 Å for contact definitions; results are **not invariant** to changing \(R_{\max}\) without re-freezing the benchmark).

### 4. Clique filtration and persistence (mathematics ↔ implementation)

**Implementation:** `src/protein/clique_persistence.py` builds a **Gudhi `SimplexTree`**: vertices at filtration 0, edges inserted at their filtration value, expansion to dimension `max_dimension` (smoke default **1** → **H0, H1**).

**Mathematics:** for each dimension you obtain a **multiset of intervals** \((b,d)\) with \(0 \le b \le d < \infty\) (after replacing \(d=+\infty\) with a finite surrogate where needed). **H0** captures connected-component scale (when components merge); **H1** captures loop-scale features from the clique complex of this filtration.

**Cached artifact:** NPZ via `topology_cache.save_topology_npz` / `load_topology_npz`: stores `coords`, `edges_*`, `persistence` array \((\mathrm{dim}, b, d)\), policy id, mmCIF hash, `graph_mode`, `max_dimension`.

### 5. End-to-end smoke pipeline (Layer A → B → C)

**Driver:** `scripts/protein/run_smoke_pipeline.py` (`PROJECT_ROOT` on `sys.path`). **ColabFold + GPU:** run from **WSL2** with `bash scripts/protein/wsl_run_smoke_from_windows_repo.sh` when the repo is on `/mnt/c/...` and the venv lives under `$HOME/...` (see script header).

| Layer | Role | Main outputs | Code |
|-------|------|--------------|------|
| **A** | Parse mmCIF; infer **primary chain**; count residues; hash file | `layer_a_structure.csv` | `mmcif_io`, `infer_primary_chain_id`, corpus glob `*.cif` |
| **B** | Build / load **topology NPZ**; edge and **interval counts** | `layer_b_topology.csv`, `data/processed/protein/*.npz` | `_ensure_topology_npz`, `build_edges_and_persistence`, `default_topology_npz_path` (keyed by policy + path + chain) |
| **C** | **ColabFold** FASTA + `colabfold_batch` → `predictions/*.pdb` | `layer_c_colabfold_smoke.csv`, `colabfold_batch_stdout_stderr.log`, `smoke_cuda_validation.json` | `run_colabfold_predict_smoke.py` |

**Figures (matplotlib, transparent PNG, DPI 300):** `smoke_layer_a_b_counts.png` (residue vs edge counts), `smoke_persistence_by_dim.png` (stacked H0/H1 interval counts per structure).

**3D persistence diagrams:** first `max_panels` structures → `figures/smoke/assets/<PDB>_pd_3d.png` via `smoke_persistence_3d.render_npz_persistence_3d` → `scripts/figures/persistence_diagram_3d_engine.py` (bars over birth–death × multiplicity).

**Residue graph viz:** PCA of cached **coords** (first 2 singular vectors); edges as faint segments; subsample at most **8000** edges if huge; RNG **seed 14** for subsample → `smoke_residue_graph_pca.png`.

**Structure views (optional):** py3Dmol + Playwright `render_structure_cartoon_surface.py` → `structure_views/*_cartoon.png`, `*_surface.png`.

**Manifest:** `smoke_artifacts_manifest.csv` lists every table/figure path and a one-line description.

### 6. ColabFold inference (environment and subprocess contract)

**Implementation:** `run_colabfold_predict_smoke.py` builds **one FASTA** from Layer A (up to `--colabfold-max-structures`), runs `colabfold_batch FASTA out/predictions`.

**GPU gate:** `ensure_nvidia_gpu_present` (nvidia-smi-based) before batch.

**Critical environment (`_colabfold_batch_env`):** `PYTHONNOUSERSITE=1`, **strip `PYTHONPATH` and `PYTHONHOME`** so user-site **JAX** (`~/.local`) cannot shadow Pixi; plus LocalColabFold-style **XLA / memory** env vars (`_wsl_jax_env`). Avoids PJRT version mismatch and import chaos.

**Defaults:** often **public ColabFold MSA server** (fair-use); optional `--colabfold-extra-args` for VRAM (e.g. `--max-msa`, `--num-recycle`).

**Prediction selection for metrics:** `find_colabfold_ranked_pdb` prefers filenames containing `ranked_001` / `ranked_1`, else any PDB whose stem contains the **PDB id**.

### 7. Per-structure smoke metrics (global geometry)

**File:** `compute_per_structure_metrics` in `src/protein/smoke_metrics.py`.

- Loads **native CA** from mmCIF (given chain).
- Finds prediction PDB; loads **pred CA** and **B-factors** (interpreted as **pLDDT** if scale looks like 0–100 or fraction).
- If length mismatch: **truncate to \(\min(n_{\mathrm{nat}}, n_{\mathrm{pred}})\)** and note in `metric_notes`.
- **Kabsch RMSD:** centered `pred` and `native`, SVD of cross-covariance, optimal rotation (proper: reflect last singular vector if \(\det(R)<0\)), then \(\mathrm{RMSD} = \sqrt{\frac{1}{n}\sum_i \| \hat{x}_i - R x_i \|^2}\) in **Å**.
- **Persistence summaries on native NPZ:** sums of **finite** \((d-b)\) for H0 and H1 (“total persistence mass” scalars).

These rows are **one line per PDB** in `smoke_per_structure_metrics.csv` and are **independent** of the nine-cell grid except they share the same prediction file when present.

### 8. Virtual persistence and RKHS loss (implementation detail)

**Vectorization:** `gudhi_persistence_to_vpd_vector` / `persistence_diagram_to_vpd_vector` in `src/topology/vpd.py`: fix a **grid_size × grid_size** histogram over \((\mathrm{birth}, \mathrm{death})\) with bins from **pooled** native+pred ranges (per homology dimension); each interval increments the count in one cell → length **\(G^2\)** integer vector.

**Virtual difference:** `virtual_difference_vector(v^{\mathrm{pred}}, v^{\mathrm{nat}})` = **integer** multiplicity difference (signed multiplicities on histogram cells, in the spirit of **virtual / Grothendieck-style** diagram differences used in `src/topology/vpd.py`).

**RKHS surrogate:** `TopologicalRKHSLoss(vdiff)` embeds `vdiff` through **RFF** derived from the **heat kernel** multiplier on the torus (`heat_multiplier`, `laplacian_symbol`), then forms a **non-negative** loss \(2(k(0) - k(\gamma,0))\)-type score (see `loss.py`). **Not** a full Gaussian RKHS Gram inversion — it is the **Monte-Fourier** / RFF approximation used in the epidemiology track.

**Randomness:** `TopologicalRKHSLoss(..., random_state=...)` fixes \(\omega\), \(\beta\) in RFF; smoke nine-grid uses **level-dependent seeds** so rows are reproducible given fixed code and data.

### 9. Nine-cell experiment (modules, stages, and exact cell formulas)

**Implementation:** `src/protein/smoke_nine_experiments.py`. **Table figure:** `scripts/protein/smoke_table_render.py` — headers **Level**, **Stage**, **TM-score**, **pLDDT**.

**Module axis (Level 0,1,2):** partition residue indices **\(0,\dots,N-1\)** into **three contiguous blocks** with `numpy.array_split` (as equal as possible). This is a **minimal** instantiation of **modules** \(P_a\); production work can swap in **domains**, **secondary-structure runs**, or **graph neighborhoods** without changing the **stage** logic.

**TM-score (CASP-style):** on each module \(P_a\), **CA-only** coordinates \(\hat{X}\), \(X^\ast\) restricted to indices in \(P_a\) are passed to `src/protein/tm_score_eval.py`. **Primary:** run **TM-align** (`TMalign` on `PATH` or `TMALIGN_BIN`) on minimal CA PDBs — the same tool family used in CASP-style structure comparison. **Fallback:** Zhang–Skolnick TM-score after Kabsch alignment when TM-align is unavailable. Install TM-align on Linux/WSL: `scripts/protein/install_tm_align_linux.sh`.

**Stages (Baseline / Wasserstein / RKHS):** these labels name **which pLDDT summary** is shown in the second column; they do **not** select different predictions in smoke today. **TM-score is identical** across the three stages at a given level (one ColabFold output per structure) until separate stage checkpoints exist.

- **Baseline — pLDDT:** **mean** of per-residue pLDDT in the module.
- **Wasserstein — pLDDT:** **minimum** pLDDT in the module.
- **RKHS — pLDDT:** **sample standard deviation** of pLDDT in the module.

**Primary structure for the nine-grid:** code uses **`df_a.iloc[0]`** for mmCIF path + chain and the **first merged** per-structure row’s prediction path (first PDB in Layer A with a complete A+B join). All nine numbers refer to **that one** native–predicted pair.

**Relation to theory:** the nine-cell table is a **diagnostic layout** (level × stage names); it no longer reports \(W_2\) or RKHS loss values. Full **Wasserstein / RKHS training** objectives and internal \(\mathcal{C}(P)\) virtual-diagram families (**DEF-S01**) belong to the **fine-tuning** story, not this smoke grid. Interval-of-interval objects \(D^{(n)}\) and pooling over internal pairs are **out of scope** for the nine-cell CSV/PNG.

### 10. Statistical interpretation, limitations, and leakage

- **Modules are not i.i.d.:** the three thirds are **spatially ordered** contiguous blocks; they are **dependent** (shared fold context). The nine cells are **descriptive diagnostics**, not nine independent experiments in the inferential sense.
- **pLDDT statistics** (mean/min/std) are **functional summaries** of the **model’s** confidence on the **prediction**; they are **not** native “quality.”
- **TM-score** compares **prediction vs native** CA geometry on the **same** module (CASP-comparable when TM-align runs). **pLDDT** summarizes the **model’s** confidence on the prediction, not native quality.
- **Selection:** smoke often uses **max-structures 1** — metrics reflect **one** ColabFold run (stochastic MSA, model choice, seed policies upstream). Reproducibility requires pinning **ColabFold / JAX** versions and MSA mode where possible.
- **Evaluation leakage (future CATH work):** official splits must use **superfamily** (and optional S35) as frozen in `dataset_policy` / manifest docs; smoke on a handful of mmCIFs does **not** replace that protocol.

### 11. WSL sync, filesystem, and results mirroring

- **`wsl_sync_minimal_to_home.sh`:** copies a **minimal** tree from a repo on **`/mnt/c/...`** to **`$HOME/<repo-folder-name>`** (override with **`WSL_HOME_REPO`**): `src/protein`, `src/topology`, `scripts/protein`, `scripts/figures/persistence_diagram_engine.py`, `scripts/figures/persistence_diagram_3d_engine.py`, requirements, mmCIF inputs. Records **Windows** repo path in `.wsl_windows_repo` for reverse sync.
- **`wsl_run_smoke.sh`:** refuses running on a repo rooted on `/mnt/c` (performance / symlink issues); activates **`.venv-wsl-gpu`**; runs `run_smoke_pipeline.py`; optionally **`wsl_sync_results_to_windows.sh`** rsync `results/protein/` back unless `WSL_SKIP_RESULTS_SYNC=1`.

### 12. Module map (quick reference)

| Concern | Path(s) |
|--------|---------|
| Policy + radii | `src/protein/dataset_policy.py` |
| Graph + PH | `residue_graph.py`, `clique_persistence.py`, `topology_cache.py` |
| Smoke driver | `scripts/protein/run_smoke_pipeline.py` |
| ColabFold smoke from a **/mnt/c/...** repo using the WSL **`.venv-wsl-gpu`** (same files visible in Windows) | `scripts/protein/wsl_run_smoke_from_windows_repo.sh` |
| ColabFold CLI | `scripts/protein/run_colabfold_predict_smoke.py` |
| Global RMSD/plDDT/NPZ sums | `src/protein/smoke_metrics.py` |
| Nine-cell metrics + TM-score | `src/protein/smoke_nine_experiments.py`, `src/protein/tm_score_eval.py` |
| TM-align (Linux/WSL install) | `scripts/protein/install_tm_align_linux.sh` |
| Table PNG | `scripts/protein/smoke_table_render.py` |
| RKHS / RFF | `src/topology/loss.py`, `src/topology/kernels.py`, `src/topology/rkhs_torch.py`, `src/topology/vpd.py` |
| Training contract (scaffold) | `src/protein/training_contract.py`, `scripts/protein/run_training_contract_demo.py` |
| Alignment lint | `scripts/protein/verify_protein_alignment.py` |
| One-shot CI (lint + topology smoke + RKHS grad demo; add `--with-smoke` for Layers A+B PNG/CSVs) | `scripts/protein/run_protein_ci_checks.py` |
| Legacy (non-ColabFold) | `scripts/protein/legacy/`, `src/protein/legacy/` |
| 3D PD | `scripts/protein/smoke_persistence_3d.py`, `scripts/figures/persistence_diagram_3d_engine.py` |

---

## Next actions
- Epidemiology: continue training and figure regeneration from frozen caches as needed (`scripts/run_collective_benchmark.py`, tuning scripts).
- Protein: keep the topology cache **model-agnostic**; ColabFold baseline via LocalColabFold + manifest \(\rightarrow\) FASTA \(\rightarrow\) predictions; then **fine-tune** with **one** competing topology objective at a time (Wasserstein **or** RKHS recipe vs baseline — not stacked for primary comparisons; see Design freeze). Meanwhile: PDB passes, CATH merge, parse/graph enrichment columns, superfamily split tables.

## Known issues / cautions
- Epi: verify `Y_t` spread before large training runs; avoid temporal leakage.
- Protein: large raw PDB mirrors stay outside git; keep validation-report filters for real benchmarks.
