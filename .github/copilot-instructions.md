# Copilot instructions — Melting Point Modeling

Purpose
- Help an AI coding agent be immediately productive in this repo: how data flows, where to look, and the concrete commands used by humans.

Big picture (short)
- Data flow: raw CSVs in `main-data/` -> feature extraction notebook (`work/version-0/content/preprocessing.ipynb`) -> feature artifact `result/data/melting_point_features.csv` -> analysis & baseline model in `work/version-0/content/eda.ipynb` -> outputs saved to `work/.../result/data/`.
- Notebooks are the primary executable units (feature engineering + EDA + baseline training). Small helper scripts (e.g., `gotpub.py`) perform data collection/ingestion.

Key files to read first
- [README.md](README.md): high-level description and reproducible nbconvert commands.
- [gotpub.py](gotpub.py): PubChem fetch script; demonstrates resume behavior (reads `main-data/melting_point_results.csv`) and CLI `--debug`/`-d` flag.
- [work/version-0/content/preprocessing.ipynb](work/version-0/content/preprocessing.ipynb): feature extraction (RDKit descriptors, fingerprints, occasional 3D features).
- [work/version-0/content/eda.ipynb](work/version-0/content/eda.ipynb): cleaning, median imputation, LightGBM baseline training, metrics and prediction outputs.
- Example artifact: [work/version-0.5/result/data/melting_point_features.csv](work/version-0.5/result/data/melting_point_features.csv)

Project conventions & patterns (concrete)
- Notebooks are treated as executable pipelines. Use `jupyter nbconvert --to notebook --execute <notebook>` to reproduce steps (see `README.md` for examples).
- Feature CSVs may contain fingerprint bitstrings and heavy non-numeric columns; `eda.ipynb` drops or converts these before modeling. Look for columns named like `morgan_fingerprint_bits`, `maccs_keys_bits`.
- Many descriptors are 3D-derived; if 3D embedding/optimization fails, those columns will be missing or NaN — downstream code fills with medians.
- Output paths are stable: notebook outputs are saved under `work/.../result/data/` — prefer writing to the same structure to stay consistent with existing notebooks.

Environment & commands (how humans run things)
- Conda recommended for RDKit: create `mp_env` and install from conda-forge.
  ```bash
  conda create -n mp_env python=3.10 -y
  conda activate mp_env
  conda install -c conda-forge rdkit pandas numpy scikit-learn matplotlib lightgbm -y
  ```
- Reproduce feature extraction and EDA (from repo root):
  ```bash
  jupyter nbconvert --to notebook --execute work/version-0/content/preprocessing.ipynb --ExecutePreprocessor.timeout=600
  jupyter nbconvert --to notebook --execute work/version-0/content/eda.ipynb --ExecutePreprocessor.timeout=600
  ```
- Run PubChem fetch script (resumes using `main-data/melting_point_results.csv`):
  ```bash
  python gotpub.py           # normal
  python gotpub.py -d        # debug mode
  ```

What to change and how (AI-focused guidance)
- If changing descriptors, update the `preprocessing.ipynb` cells and re-run via nbconvert to regenerate `result/data/melting_point_features.csv` for downstream EDA.
- Avoid editing `eda.ipynb` outputs directly — change logic in the notebook cells that construct the DataFrame, then re-run to regenerate artifacts.
- When adding a new reproducible script (non-notebook), place it at repo root or under `scripts/` and update `README.md` with the execution example.

Edge cases & gotchas (from reading code)
- `gotpub.py` rate-limits and resumes by reading the existing `main-data/melting_point_results.csv`. Be careful when parallelizing fetches — the script assumes sequential resume.
- 3D features may be missing for a subset of compounds; downstream notebook behavior is to impute medians, not drop rows.
- Fingerprints are stored as long bitstring columns — modeling notebooks currently drop or reduce these before fitting LightGBM.

When in doubt
- Start with `README.md` and run the two nbconvert commands to reproduce the pipeline end-to-end; inspect `work/.../result/data/` for produced artifacts and examples.

If any of these paths are outdated for your working branch, tell me which files changed and I will merge/update this guidance.
