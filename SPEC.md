# SPEC — RLD-to-PSD Random Vibration Fatigue & Resonance Predictor

## 1. Goal

Build an end-to-end pipeline that converts multi-axis vehicle road-load-data (RLD)
accelerometer signals into PSD spectral matrices, computes physics-based fatigue damage
as ground truth, and trains ML models that predict, from the PSD alone:

1. **Fatigue life** `T` in **hours** (to Miner damage `D = 1`).
2. **Dominant resonance frequency** `f_res` in **Hz**.

Results are explored in an interactive Streamlit + Plotly dashboard that also accepts
user-uploaded CSV signals.

Engineering conventions and invariants are defined in [`CLAUDE.md`](CLAUDE.md) and are
normative for this spec.

## 2. Pipeline overview

```
duty-cycle generator ──► x(t) [3 axes] ──► Welch ──► S_xx(f) / S(f) 3×3 matrix
                                                      │
                          stress transfer H(f) ◄──────┘
                                   │
                          Dirlik PDF + Basquin S–N + Miner ──► D_rate, T_hours (truth)
                                   │
          log-PSD features ──► SpectralRidge  ┐
          log-PSD curves   ──► PSDNet1D (CNN) ┴─► T̂_hours, f̂_res ──► benchmark ──► app.py
```

## 3. `src/signal_generator.py`

### 3.1 Dataset

- **1,200 duty cycles** total, 400 per class (balanced), seeded and reproducible.
- **Axes**: X (longitudinal), Y (lateral), Z (vertical) accelerometer channels.
- Default sampling: `fs = 2048 Hz`, duration `T_cycle = 20 s` (configurable in `config.py`).

### 3.2 Duty-cycle classes

| Class | Excitation model | Characteristics |
|---|---|---|
| `rough_road` | Broadband filtered Gaussian noise (ISO 8608 class C–E style road profile × speed) | High RMS, wide band 5–300 Hz, γ ≈ 0.4–0.7 |
| `pothole_impact` | Low-level road noise + 1–5 half-sine / decaying impulses | Transient, non-stationary, high crest factor, rings at structural modes |
| `highway_cruise` | Smooth-road noise + narrowband tonal content (wheel/driveline orders) | Low RMS, narrowband peaks, γ → 1 |

Each signal passes through a randomised structural response: 1–3 SDOF/MDOF resonances with
natural frequency `f_n ∈ [10, 250] Hz` and damping `ζ ∈ [0.02, 0.08]`. Cross-axis coupling
uses a random 3×3 mixing matrix so the spectral matrix has non-trivial cross-spectra.
Per-cycle parameters (speed, road class, `f_n`, `ζ`, impulse count/amplitude, gain) are
sampled from documented ranges and stored as metadata.

### 3.3 PSD computation

- Auto-spectra via `scipy.signal.welch` (Hann, `nperseg = 1024`, 50 % overlap,
  `scaling="density"`) → `Δf = 2 Hz`, bins `0 … fs/2`.
- Cross-spectra via `scipy.signal.csd` with identical parameters → Hermitian
  3×3 spectral matrix `S(f)` per cycle.
- Model input band: **5–500 Hz** (configurable), resampled to a fixed bin count
  (default 256) on the original linear grid.

### 3.4 Ground-truth fatigue

Implemented in `src/fatigue.py`, called by the generator:

1. Stress PSD: `G_σ(f) = |H(f)|² · G_zz(f)` (critical-axis / vertical by default; the
   dominant-axis choice is configurable), `H` = documented accel-to-stress gain in MPa/g.
2. Spectral moments `m0, m1, m2, m4`; `γ`, `E[P]`.
3. **Dirlik** rainflow-range PDF `p(S)`.
4. **Basquin** S–N: `N(S) = C · S^(-k)`, defaults `k = 5`, `C` calibrated so that typical
   lives span ~10¹–10⁶ hours across the dataset.
5. **Miner** damage rate: `D_rate = E[P] · ∫ p(S) / N(S) dS` [1/s].
6. Life: `T_hours = 1 / (3600 · D_rate)`.
7. Also record narrowband (Rayleigh) damage for cross-checking.
8. Ground-truth resonance: `f_res = argmax G_σ(f)` within the structural band; also store
   the true sampled `f_n` of the dominant mode.

### 3.5 Outputs

| File | Contents |
|---|---|
| `data/signals.npz` | `x` `(N, 3, n_samples)` float32, `fs` |
| `data/psd.npz` | `freqs` `(F,)`, `psd` `(N, 3, F)`, `csd` `(N, 3, 3, F)` complex64 |
| `data/labels.csv` | `cycle_id, duty_class, rms_x/y/z, m0, m2, m4, gamma, peak_rate, damage_rate, life_hours, life_hours_narrowband, f_res_hz, f_n_true_hz, zeta, seed, ...` |

CLI: `python -m src.signal_generator --n-cycles 1200 --seed 42 --out data/`.

## 4. `src/model.py` & `src/train.py`

### 4.1 Targets & split

- Targets: `y1 = log10(life_hours)`, `y2 = f_res_hz` (optionally scaled to [0, 1] over band).
- Stratified split by `duty_class`: 70 / 15 / 15 train / val / test, fixed seed.
- Inputs standardised with statistics from the **training set only**.

### 4.2 `SpectralRidge` (baseline)

- Features: `log10` PSD averaged into ~24 log-spaced bands per axis, plus spectral
  moments (`log m0, log m2, log m4`, `γ`) → feature vector.
- Two independent Ridge regressors (or one multi-output), α selected on validation set.
- Resonance option: Ridge regression, compared against a trivial "argmax of input PSD"
  heuristic.

### 4.3 `PSDNet1D` (PyTorch 1D-CNN)

- Input: `(batch, 3, F)` log10-PSD curves (channels = axes), optionally + 3 cross-spectral
  coherence channels.
- Architecture (default): 4 × [Conv1d(k=7) → BatchNorm → GELU → MaxPool(2)], channels
  32→64→128→128, global average pooling, MLP head (128→64→2).
- Two heads: `log10(life)` regression and `f_res` regression.
- Loss: `L = MSE(log life) + λ · SmoothL1(f_res_scaled)`, `λ = 1.0` default.
- Optimiser: AdamW (lr 1e-3, wd 1e-4), cosine schedule, ≤ 100 epochs, early stopping
  on validation loss (patience 10). Deterministic seeds; CPU-trainable in < 5 min.

### 4.4 Benchmark (`src/train.py`)

Metrics on the held-out test set, overall and per `duty_class`:

| Target | Metrics |
|---|---|
| Fatigue life | RMSE & MAE on `log10(hours)`, R², % of predictions within factor 2 and factor 10 of truth |
| Resonance | MAE (Hz), MAPE (%), % within ±1 bin (`±Δf`) |

Outputs: `artifacts/ridge.pkl`, `artifacts/psdnet1d.pt`, `artifacts/scalers.pkl`,
`artifacts/metrics.csv`, `artifacts/metrics.md` (side-by-side table), and
predicted-vs-true data for the dashboard.

**Acceptance targets** (on default synthetic data): CNN life R² ≥ 0.90 and ≥ 90 %
within factor 2; CNN resonance MAE ≤ 5 Hz; CNN matches or beats Ridge on both targets.

CLI: `python -m src.train --data data/ --out artifacts/ [--epochs N] [--seed S]`.

## 5. `app.py` — Streamlit + Plotly dashboard

### 5.1 Inputs (sidebar)

- **Source**: pick a generated cycle (filter by duty class / cycle id) **or** upload a CSV.
- **CSV upload**: columns `time` (s) + one to three of `ax, ay, az` (g). Sampling rate
  inferred from `time` (must be uniform within 1 %) or entered manually. Validation errors
  are shown with `st.error`, never a stack trace.
- Model selector: `SpectralRidge` / `PSDNet1D` / both.
- S–N parameters (`k`, `C`) and stress gain override for "what-if" ground-truth recompute.

### 5.2 Panels

1. **Time-domain signal** — Plotly line chart per axis, with RMS, peak and crest factor.
2. **PSD frequency spectrum** — log-log Welch PSD per axis, detected resonance peaks
   marked (`scipy.signal.find_peaks` on the log-PSD), predicted `f̂_res` as a vertical line,
   true `f_res` when available.
3. **Fatigue prediction** — KPI tiles: predicted life (hours, with readable units
   e.g. "3.2 × 10⁴ h"), Dirlik ground-truth life (computed live from the PSD), ratio
   predicted/true, dominant resonance (Hz).
4. **Model benchmark** — metrics table from `artifacts/metrics.csv` and predicted-vs-true
   scatter (log axes, factor-2 band).
5. **Download** — computed PSD and predictions as CSV.

### 5.3 Behaviour

- Artifacts loaded once with `st.cache_resource`; data with `st.cache_data`.
- If artifacts are missing, show instructions to run the generator and trainer.
- Uploaded signals use the exact same `src/` preprocessing as training (shared code path).

Run: `streamlit run app.py`.

## 6. Testing (`tests/`, Pytest)

- **Generator**: shapes, dtypes, class balance, determinism with fixed seed.
- **Welch**: Parseval (`∫PSD df ≈ var(x)`, ±5 %); sine of known frequency peaks at the
  right bin; spectral matrix Hermitian.
- **Fatigue**: Dirlik PDF ≥ 0 and integrates to 1; narrowband limit (γ → 1) approaches
  Rayleigh damage; PSD scaled by `a` → life scales by `a^(-k/2)`; life > 0 and finite.
- **Models**: forward pass shapes; one-batch overfit test; saved/loaded model gives
  identical outputs; predictions positive life and in-band resonance.
- **App I/O**: CSV parser accepts valid files, rejects non-uniform time / missing columns.

## 7. Non-goals

- Real vehicle data ingestion beyond CSV upload; multiaxial critical-plane fatigue;
  non-Gaussian / kurtosis corrections (may be listed as future work); GPU requirement.

## 8. Deliverables checklist

- [ ] `src/config.py`, `src/signal_generator.py`, `src/fatigue.py`, `src/features.py`
- [ ] `src/model.py`, `src/train.py`, `src/predict.py`
- [ ] `app.py`
- [ ] `tests/` passing with `pytest -q`
- [ ] `requirements.txt`, `.gitignore` (`.venv/`, `data/`, `artifacts/`)
- [ ] `artifacts/metrics.md` benchmark table committed to README or PR description
