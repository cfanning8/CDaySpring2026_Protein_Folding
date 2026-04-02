# C-DAY: Epidemiology and Protein Folding

## Current status
- Two research tracks share `src/topology/` (persistence, virtual diagrams, RKHS-style losses). That code originated with the epidemiology project; for proteins it is **only** a library for optional **auxiliary loss terms**, not a graph encoder or temporal model.
- Epidemiology: mature TGN + constraint + RKHS scripts under `scripts/`; outputs in `results/output/` and `results/figures/`.
- Protein folding: scaffold under `src/protein/` and `scripts/protein/`; **predictive baseline is ColabFold** (AlphaFold2-class via `colabfold_batch`, not TGN). PDB/mmCIF inputs go under `data/protein/mmcif/` (tracked in this handoff repo). LocalColabFold installs separately from the project venv (see protein smoke section).

## GitHub handoff & automation log (Spring 2026)

This tree is intended to match **[cfanning8/CDaySpring2026_Protein_Folding](https://github.com/cfanning8/CDaySpring2026_Protein_Folding)**. The sections below record what was built in-editor versus what you must run on a GPU machine.

### Implemented in this repository (code + docs)
- **Smoke pipeline:** Layers 1–2 (structure + topology NPZ/tables/figures) and Layer 3 hook for **ColabFold** via `colabfold_batch` (`scripts/protein/run_smoke_pipeline.py`, `scripts/protein/run_colabfold_predict_smoke.py`). Legacy OpenFold3 scripts remain optional and are not the default driver.
- **WSL2 workflow:** minimal copy to ext4 (`wsl_sync_minimal_to_home.sh`), project venv (`wsl_setup_protein_venv.sh`), LocalColabFold installer stub (`wsl_install_localcolabfold.sh`), smoke runner (`wsl_run_smoke.sh`), env probe (`wsl_probe_env.sh`), torch repair (`wsl_repair_torch_cu124.sh` with `SKIP_OPENFOLD3_REPAIR=1` for ColabFold-only).
- **Guards / probes:** `src/protein/cuda_guard.py` (torch for project venv; `nvidia-smi`-only path for ColabFold subprocess).
- **README:** ColabFold vs huge local DB, `nvcc` vs `nvidia-smi`, WSL symlink / case-sensitivity, low-VRAM flags via `--colabfold-extra-args`.

### Not completed inside the editor (you run on hardware)
- **LocalColabFold install:** long `pixi install && pixi run setup` under `$HOME/localcolabfold` — run `bash scripts/protein/wsl_install_localcolabfold.sh` in **WSL2 on ext4**, not under `/mnt/c/...`.
- **Full Layer 3 inference:** requires `colabfold_batch` on `PATH` and a working **CUDA/JAX** stack; if JAX fails, install the **CUDA toolkit** in WSL so `nvcc --version` aligns with [LocalColabFold](https://github.com/YoshitakaMo/localcolabfold) expectations (driver alone is insufficient for some errors).
- **First** `colabfold_batch` **run** downloads AlphaFold weights (automatic per upstream).

### Clone → end-to-end checklist (condensed)
1. Clone this repo; create a Python **3.10+** venv at repo root if working on Linux/native (optional on Windows for Layers 1–2 only).
2. **WSL2 (recommended for Layer 3):** `bash scripts/protein/wsl_sync_minimal_to_home.sh` from the Windows-mounted copy, then `cd ~/2_Protein_Folding` (or `WSL_HOME_REPO`).
3. `bash scripts/protein/wsl_setup_protein_venv.sh` → `bash scripts/protein/wsl_install_localcolabfold.sh`.
4. `bash scripts/protein/wsl_run_smoke.sh --skip-structure-figures`  
   Add `--colabfold-extra-args '...'` if VRAM is tight (see smoke section above).

### What is excluded from git (by design)
- Virtualenvs: `.venv/`, `.venv-wsl-gpu/`, etc.
- Cursor local files: `.cursorrules`, `.cursor/`
- Large regenerated caches: `data/processed/protein/*.npz`
- Everything else in this handoff—including **smoke mmCIF under `data/protein/mmcif/`** and **results under `results/protein/`** for figures/tables—is meant to be versioned unless a file exceeds GitHub’s size limits; split or Git LFS if a future artifact is too large.

## Repository map
- `src/` : `topology/` (shared math for persistence and losses), `models/` (**epidemiology only**: TGN stack), epidemiology loaders and simulators, `protein/` (**OpenFold-facing** data and topology preprocessing; no TGN).
- `scripts/` : epidemiology entry points at top level; protein folding under `scripts/protein/`.
- `data/epidemiology/` : optional home for epi inputs separate from legacy `data/preprocessed/` paths used by existing scripts.
- `data/protein/mmcif/` : deposited entry mmCIF files (not biological assemblies; tracked in this handoff repo).
- `data/protein/validation/` : validation PDFs paired to entries (gitignored blobs).
- `data/processed/protein/` : cached topology NPZs (coords, edge table, persistence arrays; `*.npz` gitignored).
- `results/output/` and `results/figures/` : epidemiology runs.
- `results/protein/output/` and `results/protein/figures/` : protein runs.
- `requirements-base.txt` : shared numerics/topology stack.
- `requirements-epi.txt` : default install (includes PyG and PyVista).
- `requirements-protein.txt` : Biotite, py3Dmol, Playwright for structure figures; no PyG/TGN stack (install OpenFold deps separately when training lands).

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

This track is **ColabFold / AlphaFold2-class structure prediction + cached clique-persistence topology**, not “a protein TGN.” The epidemiology project is a **historical analogy** only (both projects ask when topology-informed objectives help); the protein **structure baseline** is **ColabFold** (`colabfold_batch`), with topology entering as **precomputed native features** and/or **auxiliary losses on predicted versus native coordinates**. There is **no** planned fusion by concatenating topology vectors into the folding trunk; the target is **additive loss terms** on top of the standard folding objective.

### Layers (frozen conceptually)
1. **Structural data:** mmCIF parsing, chain/domain choice, residue representatives (see topology policy), metadata, manifests, caching keys (`mmcif` hash + topology policy id).
2. **Topology preprocessing:** weighted residue graph, clique filtration, persistence, NPZ cache under `data/processed/protein/`. This graph is a **derived geometric object for PH**; it is **not** the ColabFold/AlphaFold2 model graph.
3. **Learning:** **ColabFold** inference (public MSA server by default for small jobs) on the eligible manifest; then optional fine-tuning stages compare **predicted** \(\hat{X}\) vs **native** \(X\): **(M2)** persistence / **landscape**-style loss, **(M3)** **RKHS** semimetric on virtual persistence objects, **(M4)** **jet** / higher-order multi-scale objective ported from the epidemiology topological calculus (spec TBD). Attachment point remains **after** coordinates: build the same residue graph on \(\hat{X}\), differentiate through your summary or surrogate.

### Scientific question (prediction estimand)
- Does topology-aware **auxiliary regularization** improve structure prediction from a **ColabFold / AF2-class** baseline under hard generalization (held-out CATH homologous superfamilies)?

### Planned model ladder (ColabFold / AF2-class baseline; implement incrementally)
- Data unit: single CATH domain from PDB mmCIF; splits by superfamily (S35 helpers optional); CASP-style external check later.
- **M1 (baseline):** \(L = L_{\mathrm{AF2}}\) from ColabFold only — AlphaFold2-class structural objective (smoke: **inference** via `colabfold_batch`; weights on first run).
- **M2:** add \(\lambda_{\mathrm{PH}} L_{\mathrm{PH}}\) — persistence / **topological landscape** features of \(\hat{X}\) vs \(X\) under the frozen graph policy.
- **M3:** add \(\lambda_{\mathcal{H}} L_{\mathcal{H}}\) — **RKHS** semimetric on virtual persistence (reuse `src/topology/` kernels where applicable).
- **M4:** add jet-style / higher-order multi-scale regularizer from the **epidemiology** track’s topological calculus (implementation TBD after M2/M3).
- Primary metrics: GDT-TS; co-primary: TM-score and lDDT.

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
  2. `cd "$HOME/2_Protein_Folding"` (or `WSL_HOME_REPO`), then `bash scripts/protein/wsl_setup_protein_venv.sh` (project venv: torch **cu124**, topology + figures). If torch stack was wrong: `SKIP_OPENFOLD3_REPAIR=1 bash scripts/protein/wsl_repair_torch_cu124.sh` (ColabFold-only) or `bash scripts/protein/wsl_repair_torch_cu124.sh` (also reinstalls optional OpenFold3 deps).
  3. `bash scripts/protein/wsl_install_localcolabfold.sh` — installs **ColabFold** under `$HOME/localcolabfold` (or `LOCALCOLABFOLD_ROOT`). AlphaFold weights download on **first** `colabfold_batch` run.
  4. `bash scripts/protein/wsl_run_smoke.sh --skip-structure-figures` (prepends LocalColabFold `bin` to `PATH` when present).
  - **GPU / CUDA (LocalColabFold upstream):** GPU JAX builds expect a **CUDA toolkit** compatible with their stack (upstream docs: **CUDA ≥ 12.1**, cudnn; **CUDA 12.4** often recommended). Check the **compiler** with `nvcc --version` inside WSL — the **driver** reported by `nvidia-smi` is related but not the same thing. If you see `CUDA_ERROR_ILLEGAL_ADDRESS` or similar, fix toolkit/driver alignment before blaming the smoke scripts.
  - **MSA / databases:** Default small-batch flow uses the **public ColabFold MSA server** (no local **~940 GB** DB). Offline / fully local search is optional and separate ([`setup_databases.sh` in ColabFold](https://github.com/sokrypton/ColabFold/blob/main/setup_databases.sh)).
  - **WSL symlink issues:** If `pixi install` / setup fails on symlinks under a Windows-backed tree, install only under **`$HOME` (ext4)** (what `wsl_install_localcolabfold.sh` enforces), or per [LocalColabFold](https://github.com/YoshitakaMo/localcolabfold) enable **case-sensitive** NTFS for that folder from **Windows PowerShell**: `fsutil file SetCaseSensitiveInfo <path> enable` (not from WSL).
  - Layer 3 smoke checks **nvidia-smi** (GPU present) before `colabfold_batch` (`src/protein/cuda_guard.py`); **no CPU fallback** for structure inference in this smoke.
  - **Low VRAM / laptop GPUs:** pass tuning flags via `--colabfold-extra-args` (forwarded to `colabfold_batch`), e.g. smaller MSA caps or fewer recycles — see `colabfold_batch --help` and upstream **Flags** (e.g. `--max-msa`, `--num-recycle`).
  - GPU metadata: `results/protein/output/colabfold_smoke/smoke_cuda_validation.json`, status `layer_c_colabfold_smoke.csv`.
- Other flags: `--corpus-dir`, `--force-topology`, `--skip-structure-figures`, `--no-colabfold-smoke`, `--colabfold-dry-run`, `--colabfold-max-structures N`, `--colabfold-extra-args`, `--colabfold-allow-missing-cli`.
- **Layers 1–2:** structure tables, topology NPZ stats, matplotlib summaries.
- **Layer 3 — ColabFold:** `scripts/protein/run_colabfold_predict_smoke.py` writes FASTA, runs `colabfold_batch`. Optional legacy OpenFold3 path: `wsl_setup_openfold3_gpu.sh` + `run_openfold3_predict_smoke.py` (not wired into `run_smoke_pipeline.py` by default).
- **Layer 3 — M2–M4:** `smoke_pipeline_status.csv` lists training regimes only.
- **Outputs:** `results/protein/tables/smoke/*`, `results/protein/figures/smoke/*`, `results/protein/output/smoke_colabfold_probe.json`, `results/protein/output/colabfold_smoke/layer_c_colabfold_smoke.csv`.

### Shared reuse (math only; not architecture)
- `src/topology/` holds persistence summaries, virtual-diagram tooling, and RKHS / alignment-style losses developed for **epidemiology**. For proteins, reuse is **limited to auxiliary objectives** (and shared notation), not TGN checkpoints or temporal encoders. Clique persistence for proteins lives in `src/protein/clique_persistence.py` with filtrations matched to the residue graph policy.

## Next actions
- Epidemiology: continue training and figure regeneration from frozen caches as needed (`scripts/run_collective_benchmark.py`, tuning scripts).
- Protein: keep the topology cache **model-agnostic**; ColabFold baseline via LocalColabFold + manifest \(\rightarrow\) FASTA \(\rightarrow\) predictions; then add **additive** topology losses on \(\hat{X}\) vs \(X\) (no trunk concatenation). Meanwhile: PDB passes, CATH merge, parse/graph enrichment columns, superfamily split tables.

## Known issues / cautions
- Epi: verify `Y_t` spread before large training runs; avoid temporal leakage.
- Protein: large raw PDB mirrors stay outside git; keep validation-report filters for real benchmarks.
