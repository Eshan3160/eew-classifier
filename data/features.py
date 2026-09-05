"""
features.py
------------
Preprocessing + feature extraction for a 3-second, 3-axis acceleration window.

Extracts 20 features per window (5 metrics x [3 axes + 1 vector magnitude]):
  - PGA (Peak Ground Acceleration)
  - Signal energy
  - FFT dominant frequency
  - Pd: peak displacement in the P-wave window (integrate acc -> vel -> disp)
  - tau_c: characteristic period, tau_c = 2*pi*sqrt(integral(disp^2)/integral(vel^2))
    (Pd and tau_c are the Wu & Kanamori 2005 P-wave features used operationally
    in Japan's EEW system.)
"""


import numpy as np
from scipy.signal import butter, sosfiltfilt

FS = 100
LOWCUT, HIGHCUT = 0.5, 10.0


def bandpass_filter(window: np.ndarray, fs: int = FS, lowcut=LOWCUT, highcut=HIGHCUT, order=4):
    """Zero-phase Butterworth bandpass filter, applied per axis."""
    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    filtered = np.zeros_like(window)
    for axis in range(window.shape[1]):
        filtered[:, axis] = sosfiltfilt(sos, window[:, axis])
    return filtered


def _integrate(signal: np.ndarray, fs: int) -> np.ndarray:
    """Cumulative trapezoidal integration with linear baseline correction."""
    dt = 1.0 / fs
    integrated = np.cumsum(signal) * dt
    x = np.arange(len(integrated))
    coeffs = np.polyfit(x, integrated, 1)
    baseline = np.polyval(coeffs, x)
    return integrated - baseline


def _dominant_frequency(signal: np.ndarray, fs: int) -> float:
    freqs = np.fft.rfftfreq(len(signal), d=1 / fs)
    spec = np.abs(np.fft.rfft(signal))
    spec[0] = 0
    return float(freqs[np.argmax(spec)])


def _tau_c(disp: np.ndarray, vel: np.ndarray) -> float:
    num = np.sum(disp ** 2)
    den = np.sum(vel ** 2)
    if den <= 0:
        return 0.0
    return float(2 * np.pi * np.sqrt(num / den))


def extract_features(raw_window: np.ndarray, fs: int = FS) -> dict:
    """
    raw_window: (n_samples, 3) array, axes = [NS, EW, UD]
    Returns a flat dict of named features.
    """
    filtered = bandpass_filter(raw_window, fs=fs)
    axis_names = ["ns", "ew", "ud"]
    feats = {}

    vel_axes, disp_axes = [], []
    for i, name in enumerate(axis_names):
        acc = filtered[:, i]
        vel = _integrate(acc, fs)
        disp = _integrate(vel, fs)
        vel_axes.append(vel)
        disp_axes.append(disp)

        feats[f"pga_{name}"] = float(np.max(np.abs(acc)))
        feats[f"energy_{name}"] = float(np.sum(acc ** 2))
        feats[f"dom_freq_{name}"] = _dominant_frequency(acc, fs)
        feats[f"pd_{name}"] = float(np.max(np.abs(disp)))
        feats[f"tau_c_{name}"] = _tau_c(disp, vel)

    vec_mag = np.sqrt(np.sum(filtered ** 2, axis=1))
    vel_mag = np.sqrt(np.sum(np.stack(vel_axes, axis=1) ** 2, axis=1))
    disp_mag = np.sqrt(np.sum(np.stack(disp_axes, axis=1) ** 2, axis=1))

    feats["pga_vec"] = float(np.max(vec_mag))
    feats["energy_vec"] = float(np.sum(vec_mag ** 2))
    feats["dom_freq_vec"] = _dominant_frequency(vec_mag, fs)
    feats["pd_vec"] = float(np.max(disp_mag))
    feats["tau_c_vec"] = _tau_c(disp_mag, vel_mag)

    return feats


if __name__ == "__main__":
    from simulate import generate_event

    p_window, full = generate_event(target_pga_gal=150, seed=42)
    f = extract_features(p_window)
    for k, v in f.items():
        print(f"{k:15s} {v:.4f}")
