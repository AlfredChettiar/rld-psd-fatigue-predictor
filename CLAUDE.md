# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository. The full product
specification lives in [`SPEC.md`](SPEC.md); this file covers how to build it.

## Project in one sentence

Turn road-load-data (RLD) accelerometer time series into Power Spectral Densities (PSDs),
compute ground-truth random-vibration fatigue damage (Dirlik + Palmgren–Miner), and train
models that predict **fatigue life (hours)** and **dominant resonance frequency (Hz)**
directly from the log-PSD.

## Tech stack (pinned choices — do not substitute)

| Concern | Tool | Notes |
|---|---|---|
| Language | **Python 3.11** | Local virtualenv at `.venv/` |
| Arrays | **NumPy** | Vectorise; no Python loops over samples |
| Tables / I/O | **Pandas** | CSV load/save, metrics tables |
| Signal processing | **SciPy** | PSD via `scipy.signal.welch` **only** (no hand-rolled FFT PSDs) |
| Deep learning | **PyTorch** | 1D-CNN over log-PSD bins |
| Dashboard | **Streamlit** | `app.py` at repo root |
| Plots | **Plotly** | All interactive charts (no Matplotlib in the app) |
| Tests | **Pytest** | `tests/` mirrors `src/` |

Baseline regressor is a closed-form / scikit-style Ridge on spectral features; if
scikit-learn is added for it, record it in `requirements.txt`.

## Environment setup

```bash
python3.11 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Always run tools through the venv interpreter (`.venv/bin/python`, `.venv/bin/pytest`).
Never install packages globally.

## Common commands

```bash
python -m src.signal_generator --n-cycles 1200 --seed 42 --out data/
```

```bash
python -m src.train --data data/ --out artifacts/
```

```bash
streamlit run app.py
```

```bash
pytest -q
```

## Repository layout (modular `src/`)

```
.
├── CLAUDE.md
├── SPEC.md
├── requirements.txt
├── app.py                    # Streamlit + Plotly dashboard (UI only, no physics here)
├── src/
│   ├── __init__.py
│   ├── config.py             # Constants: fs, duration, Welch params, S-N params, seeds
│   ├── signal_generator.py   # Synthetic duty cycles, Welch PSD, ground-truth damage
│   ├── fatigue.py            # Spectral moments, Dirlik PDF, Miner damage, life
│   ├── features.py           # log-PSD transforms, band features for Ridge
│   ├── model.py              # SpectralRidge baseline + PSDNet1D (PyTorch)
│   ├── train.py              # Train/val/test split, training loops, benchmark report
│   └── predict.py            # Load artifacts, run inference on a PSD or CSV signal
├── tests/
│   ├── test_signal_generator.py
│   ├── test_fatigue.py
│   ├── test_model.py
│   └── test_app_io.py
├── data/                     # Generated datasets (git-ignored)
└── artifacts/                # Trained weights, scalers, metrics (git-ignored)
```

Rules:
- `app.py` imports from `src/`; it must not re-implement PSD, damage or model logic.
- Physics (`fatigue.py`) has no PyTorch dependency; models never compute ground truth.
- Every public function has type hints and a docstring stating **units**.
- Configuration lives in `src/config.py`; no magic numbers scattered across modules.
- Fix all randomness with explicit seeds (`numpy.random.default_rng(seed)`, `torch.manual_seed`).

## Physical fatigue-life guardrails

These are hard invariants. Code that violates them is a bug, and each one should have a test.

**Units & conventions**
- Acceleration in **g** (or m/s² — pick one in `config.py` and never mix); PSD in **g²/Hz**.
- Frequency in **Hz**, time in **s**, fatigue life reported in **hours**.
- Stress is derived from acceleration via a documented linear transfer (SDOF/FRF gain
  in `config.py`); stress PSD in **MPa²/Hz**.

**Signal processing**
- Sampling rate `fs` must satisfy Nyquist for the highest modelled resonance
  (`fs ≥ 2.5 × f_max` recommended).
- Welch: `scipy.signal.welch(x, fs=fs, window="hann", nperseg=..., noverlap=nperseg//2,
  detrend="constant", scaling="density")`, one-sided.
- **Parseval check**: `∫ PSD df ≈ var(x)` within 5 % — assert this in tests.
- PSD values must be finite and ≥ 0; clip to a small floor (e.g. `1e-12`) before `log10`.

**Spectral fatigue (Dirlik + Miner)**
- Spectral moments `m0, m1, m2, m4` computed by trapezoidal integration over the one-sided PSD.
- Irregularity factor `0 < γ = m2 / sqrt(m0·m4) ≤ 1`; peak rate `E[P] = sqrt(m4/m2)` > 0.
- Dirlik PDF must be non-negative and integrate to 1 (±1 %) over stress range.
- Basquin S–N curve `N = C · S^(-k)` with `k > 0`, `C > 0` from `config.py` (default
  steel-like `k ≈ 3–6`). Use stress **range** consistently with the S–N definition.
- Miner damage rate `D_rate ≥ 0`; damage is additive across duty-cycle blocks;
  failure at `D = 1`.
- Fatigue life `T_hours = 1 / (D_rate_per_s · 3600)`; must be positive and finite.
  Clamp only for display, never in ground truth.
- **Monotonicity sanity**: scaling the PSD by `a > 1` must *reduce* life by `≈ a^(k/2)`.
  Add this as a property test.
- Dominant resonance frequency must lie within `(0, fs/2)` and coincide with the PSD peak
  inside the modelled structural band.

**Models**
- Train on `log10(life_hours)`, not raw hours (life spans orders of magnitude).
- Model outputs must be inverse-transformed and checked: life > 0, resonance in valid band.
- Never let a model prediction feed back into ground-truth generation.
- Report metrics per duty-cycle class as well as overall.

## Coding conventions

- Format with `black` / lint with `ruff` if available; follow PEP 8 otherwise.
- Prefer pure functions; keep I/O at module edges (`__main__` blocks, `app.py`).
- Tests must be fast (< 60 s total): use small `n_cycles` and few epochs in fixtures.
- Do not commit `data/`, `artifacts/`, `.venv/`, or large binaries.
