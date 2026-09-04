"""
simulate.py
-----------
Generates synthetic 3-axis (NS / EW / UD) strong-motion accelerometer
records that mimic the general character of NIED K-NET recordings.

WHY SYNTHETIC (be upfront about this in any writeup / interview):
Real K-NET / KiK-net data requires registered download from NIED
(https://www.kyoshin.bosai.go.jp/). This generator produces physically
plausible substitutes so the pipeline can be built and demoed without
that data. See README.md for how to swap in real K-NET ASCII files later.

PHYSICAL MODEL:
- A full record = P-wave onset -> S-P gap -> S-wave (the damaging part) -> decay.
- EEW systems predict the eventual shaking from ONLY the first few seconds
  of the P-wave, before the S-wave arrives. That's the actual "early warning"
  concept: predict what's coming, not classify what already happened.
- So: we simulate a full event, but the MODEL only ever sees the first
  WINDOW_SECONDS of the P-wave. The label is the eventual JMA seismic
  intensity (shindo) computed from the full S-wave portion.

GROUND TRUTH LABELS:
Real JMA instrumental seismic intensity is NOT "PGA above some threshold."
It's computed via a specific formula (Kunugi et al.):
  1. Band-pass/period filter each component (approximated here).
  2. Vector-composite the 3 components at each time step.
  3. Find the acceleration level a0 such that the cumulative time the
     composite signal spends above a0 equals exactly 0.3 seconds.
  4. I = 2*log10(a0) + 0.94, then bucketed into the 10 JMA categories:
     0, 1, 2, 3, 4, 5-, 5+, 6-, 6+, 7
We implement this properly (jma_seismic_intensity below) so labels are
realistic, not arbitrary.
"""

import numpy as np

FS = 100 # sampling rate, Hz (matches K-NET)
WINDOW_SECONDS = 3 # the "early warning" window the model sees
WINDOW_SAMPLES = FS * WINDOW_SECONDS

# JMA shindo categories in order, used for bucketing continuous intensity I
SHINDO_LABELS = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
# Lower bound of continuous JMA intensity I for each category
SHINDO_BOUNDS = [-np.inf, 0.5, 1.5, 2.5, 3.5, 4.5, 5.0, 5.5, 6.0, 6.5]


def jma_seismic_intensity(acc_3axis: np.ndarray, fs: int = FS) -> float:
    """
    Compute the real JMA instrumental seismic intensity I from a 3-axis
    acceleration record (gal = cm/s^2), shape (n_samples, 3).
    """
    composite = np.sqrt(np.sum(acc_3axis ** 2, axis=1))

    if composite.max() <= 0:
        return 0.0

    dt = 1.0 / fs

    lo, hi = 0.0, composite.max()
    target_duration = 0.3
    for _ in range(60):
        mid = (lo + hi) / 2
        duration_above = np.sum(composite > mid) * dt
        if duration_above > target_duration:
            lo = mid
        else:
            hi = mid
    a0 = (lo + hi) / 2

    if a0 <= 0:
        return 0.0

    I = 2 * np.log10(a0) + 0.94
    return max(I, 0.0)


def intensity_to_shindo(I: float) -> str:
    """Bucket a continuous JMA intensity value into the official category."""
    idx = np.searchsorted(SHINDO_BOUNDS, I, side="right") - 1
    idx = np.clip(idx, 0, len(SHINDO_LABELS) - 1)
    return SHINDO_LABELS[idx]


def _envelope(n_samples, rise_frac, decay_rate):
    """Simple asymmetric envelope: fast rise, exponential decay."""
    t = np.arange(n_samples)
    rise_len = int(n_samples * rise_frac)
    env = np.ones(n_samples)
    if rise_len > 0:
        env[:rise_len] = np.linspace(0, 1, rise_len)
    decay_t = t - rise_len
    decay_t[decay_t < 0] = 0
    env *= np.exp(-decay_rate * decay_t / n_samples)
    return env


def generate_event(target_pga_gal: float, seed: int = None):
    """
    Simulate one full earthquake record (P-wave + S-wave) scaled to roughly
    reach `target_pga_gal` during the S-wave portion.

    Returns:
        p_window: (WINDOW_SAMPLES, 3) array -- the ONLY data the model sees
        full_record: (n_samples, 3) array -- used to compute the true label
    """
    rng = np.random.default_rng(seed)

    sp_time = rng.uniform(2.0, 15.0)
    sp_samples = int(sp_time * FS)

    total_seconds = WINDOW_SECONDS + sp_time + rng.uniform(8, 20)
    n_samples = int(total_seconds * FS)

    t = np.arange(n_samples) / FS

    def band_noise(n, f_lo, f_hi, fs):
        """Colored noise limited to a frequency band, via FFT shaping."""
        white = rng.normal(size=n)
        freqs = np.fft.rfftfreq(n, d=1 / fs)
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        spec = np.fft.rfft(white)
        spec[~mask] = 0
        return np.fft.irfft(spec, n)

    axes = []
    for axis in range(3):
        p_wave = band_noise(n_samples, 2.0, 10.0, FS)
        p_env = np.zeros(n_samples)
        p_env[:sp_samples] = _envelope(sp_samples, rise_frac=0.15, decay_rate=1.5)
        p_wave *= p_env
        p_wave *= target_pga_gal * 0.08

        s_wave = band_noise(n_samples, 0.5, 6.0, FS)
        s_len = n_samples - sp_samples
        s_env = np.zeros(n_samples)
        s_env[sp_samples:] = _envelope(s_len, rise_frac=0.1, decay_rate=2.2)
        s_wave *= s_env

        combined = p_wave + s_wave
        axes.append(combined)

    full_record = np.stack(axes, axis=1)

    s_portion = full_record[sp_samples:]
    current_pga = np.max(np.abs(s_portion)) + 1e-9
    full_record *= (target_pga_gal / current_pga)

    p_start = max(sp_samples - WINDOW_SAMPLES, 0)
    p_window = full_record[p_start:p_start + WINDOW_SAMPLES]
    if len(p_window) < WINDOW_SAMPLES:
        pad = WINDOW_SAMPLES - len(p_window)
        p_window = np.pad(p_window, ((pad, 0), (0, 0)))

    return p_window, full_record


if __name__ == "__main__":
    for pga in [5, 50, 200, 600]:
        pw, full = generate_event(pga, seed=1)
        I = jma_seismic_intensity(full)
        print(f"target_pga={pga:>4} gal | true JMA I={I:.2f} | shindo={intensity_to_shindo(I)} | "
              f"P-window shape={pw.shape}")
