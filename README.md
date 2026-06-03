# Reproduction package — Ionic-liquid / graphene-interface Na⁺ diffusion (RDF vs persistent homology)

This folder contains the data, intermediate caches, and analysis scripts needed to
reproduce the figures and machine-learning results of the manuscript
*"Physical interpretation of ionic-liquid diffusion at graphene interfaces using
persistent homology."*

The raw 100-ns LAMMPS trajectories are **not** included. Instead, the precomputed
intermediates in `cache/` let you reproduce everything downstream of the
trajectories (diffusion coefficients, descriptors, ML, inverse analysis).

---

## Directory layout

```
PH_il_analysis/
├── README.md                  ← this file
├── pyproject.toml             dependency declaration (uv / PEP 621)
├── uv.lock                    pinned dependency versions
├── .python-version            pinned Python version (3.10)
├── scripts/                   analysis scripts (Python) + original notebook
├── data/                      processed per-configuration data (CSV/XLSX)
├── results/                   machine-learning result tables (CSV) + figures
├── cache/                     precomputed intermediates (see below)
```

### scripts/
| file | role |
|------|------|
| `diffusion_ml_v3.py` | recompute cumulative diffusion coefficient `D_cumul = MSD(T)/(6T)` from raw MSD and run the baseline ML (produces `D_cumulative.csv`, `ml_results_v3.csv`). v1/v2 are earlier iterations. |
| `rdf_zdensity_analysis.py` | RDF + Na⁺ z-density descriptors and ML comparison (`rdf_ml_results.csv`). |
| `pd_vector_analysis.py` | 2nd-order (H₂) persistence-image vector: correlation maps, residual analysis, H₂ diagrams. |
| `ph_targeted_ml.py` | interface-targeted H₂ descriptor + ML (`ph_targeted_results.csv`). |
| `h1_analysis.py` | **main result**: 1st-order (H₁) persistence diagrams, H₁ vectors + targeted H₁ descriptor, ML comparison, H₁ inverse analysis (`h1_ml_results.csv`, `cache/h1_*.npy`). |
| `homcloud_inverse_analysis.py` | H₂ inverse analysis (`optimal_volume`): boundary-atom identification of the Na⁺ confinement cavity at 250 K / 450 K. |
| `homcloud_comprehensive.py` | combined visualization (per-condition H₂ maps, inverse analysis, trends). |
| `export_summary.py` | exports summary figures/tables. |
| `20211202_ML_il.ipynb` | original exploratory notebook (starting point). |

### data/
- `D_cumulative.csv` — cumulative diffusion coefficient for all **500 configurations** (100 simulations × 5 snapshots). Columns: `p_idx, sim_id, il_type, comp_idx, temp, efield, snap_ns, D_cumul`. This is the central processed dataset and the ML target source.
- `D_per_simulation.csv`, `D_recalculated.csv` — per-simulation and intermediate diffusion values.
- `il_data_gra.csv`, `il_data.xlsx` — original tabulated input data.

### results/
Machine-learning comparison tables (CV R², test R², RMSE) for each descriptor family:
`ml_results*.csv` (baseline/v1–v3), `rdf_ml_results.csv`, `ph_targeted_results.csv`,
`h1_ml_results.csv` (the H₁ results that are the paper's main finding).

### cache/  (precomputed intermediates — enable reproduction without raw trajectories)
| item | contents |
|------|----------|
| `homcloud_txt/` | 500 files (`000.txt`–`499.txt`): xyz coordinates (4,269 atoms each) of every sampled snapshot — input to all persistent-homology computations. |
| `pd_vector/` | 500 files (`p000.dat`–`p499.dat`): H₂ persistence pairs per configuration. |
| `pd_vectors.npy` | 500 × 2080 H₂ persistence-image vectors. |
| `h1_vectors.npy` | 500 × 2016 H₁ persistence-image vectors. |
| `h1_features.npy` | 500 × 6 interface-targeted H₁ descriptors (birth, death, persistence, n>1, n>2, mean-persistence). |
| `ph_targeted_features.npy` | interface-targeted H₂ descriptors. |
| `rdf_matrix.npy` | 500 × 600 Na⁺–all-atom RDF g(r). |
| `zdensity_matrix.npy` | 500 × 50 Na⁺ z-density profiles. |
| `homcloud/` | `BF4_No01_{250,450}K_100ns.data` (LAMMPS snapshots) and `.idiagram` (HomCloud persistence diagrams) used for the inverse-analysis figures. |
| `read_all.ps1` | helper script used when collecting the raw outputs. |



---

## Environment

Dependencies are pinned in [`pyproject.toml`](pyproject.toml) /
[`uv.lock`](uv.lock) and the Python version in [`.python-version`](.python-version).
We use [uv](https://docs.astral.sh/uv/) for fast, reproducible setup.

### 1. Install uv (once)

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Create the environment

```bash
git clone <this repo url>
cd PH_il_analysis
uv sync           # installs the locked Python 3.10 + all deps into .venv/
```

That's it — no separate `pip install` or `conda env create`. The locked
dependency set is:

- `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`
- `python-docx` (only needed by `scripts/export_summary.py`)
- `homcloud` (v5.3+) — persistence-diagram computation and `optimal_volume`
  inverse analysis

LAMMPS is only required to regenerate raw trajectories (step 1 below); it is
**not** needed if you use the provided `cache/` and `data/`.

---

## Reproduction pipeline

Run each step with `uv run` (this uses the locked environment without needing
to manually activate `.venv/`):

1. **Diffusion coefficients** (needs raw MSD trajectories; otherwise use
   the provided `data/D_cumulative.csv`):
   `uv run python scripts/diffusion_ml_v3.py`
2. **RDF / z-density descriptors + ML:** `uv run python scripts/rdf_zdensity_analysis.py`
3. **H₂ persistence vector analysis:** `uv run python scripts/pd_vector_analysis.py`
4. **Targeted H₂ descriptor + ML:** `uv run python scripts/ph_targeted_ml.py`
5. **H₁ analysis (main result):** `uv run python scripts/h1_analysis.py`
6. **H₂ inverse analysis (cavity):** `uv run python scripts/homcloud_inverse_analysis.py`
7. **Comprehensive figures:** `uv run python scripts/homcloud_comprehensive.py`,
   `uv run python scripts/export_summary.py`

Steps 2–7 run from the `cache/` intermediates and do **not** require the raw
trajectories.

---

## ⚠️ Path configuration

Input/output paths are resolved relative to the repository root via
`Path(__file__).resolve().parent.parent`, so the scripts work out of the box
after `git clone`. Outputs go to:

- `data/` — processed CSV intermediates (e.g. `D_cumulative.csv`)
- `results/` — ML result tables (`*.csv`) and figures (`*.png`)
- `cache/` — persistent-homology intermediates (read-only for steps 2–7)

The only path you may need to edit is `RAW_WIN_BASE` (in
`diffusion_ml_v3.py`, `rdf_zdensity_analysis.py`, `ph_targeted_ml.py`,
`h1_analysis.py`, `homcloud_inverse_analysis.py`) — the raw-trajectory
location, only used for regenerating step 1 outputs.

---

## Dataset summary

100 simulations = 2 ionic liquids (EMI–BF₄, EMI–TFSI) × 5 compositions
(Na⁺ count: BF₄ 79–159, TFSI 39–119, in steps of 20) × 5 temperatures
(250–450 K) × 2 electric-field conditions (off/on). Five snapshots per simulation
(60, 70, 80, 90, 100 ns) → 500 configurations. Every system is a graphene-interface
system (each cell contains a 960-carbon graphene layer; 4,269 atoms total).
See the Supplementary Information (Table S2) of the manuscript for the full
per-condition list.
