# Legacy / embargoed protein helpers

Per **README → Protein folding → Design freeze (DEF-O01)**, the **publication and fine-tuning path is ColabFold only**.

Files here install or call **non-ColabFold** structure stacks. They are **not** referenced by `run_smoke_pipeline.py` or topology fine-tuning code.

| Script | Role |
|--------|------|
| `run_openfold3_predict_smoke.py` | Third-party CLI smoke (historical). |
| `wsl_setup_openfold3_gpu.sh` | Optional venv stack conflicting with JAX/ColabFold. |
