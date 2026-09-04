"""
build_dataset.py
------------------
Generates a full synthetic dataset of P-wave windows -> features -> true
shindo labels, with a REALISTIC class imbalance: real earthquake catalogs
are heavily skewed toward small events, so most records are shindo 0-2
and very few are shindo 6+/7. We replicate that skew here rather than
balancing the raw data -- the imbalance itself is part of what the model
has to handle (hence class weighting at training time).
"""

import numpy as np
import pandas as pd
from simulate import generate_event, jma_seismic_intensity, intensity_to_shindo, SHINDO_LABELS
from features import extract_features

N_SAMPLES = 6000
SEED = 2026


def sample_target_pga(rng: np.random.Generator) -> float:
    """
    Sample a target S-wave PGA (gal) from a heavy-tailed distribution so
    low-intensity events dominate and high-intensity events are rare.
    """
    return float(rng.exponential(scale=25.0)) + 0.5


def build(n_samples: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_samples):
        target_pga = sample_target_pga(rng)
        event_seed = int(rng.integers(0, 2**31 - 1))
        p_window, full_record = generate_event(target_pga, seed=event_seed)

        feats = extract_features(p_window)
        true_I = jma_seismic_intensity(full_record)
        shindo = intensity_to_shindo(true_I)

        feats["true_intensity_value"] = true_I
        feats["shindo"] = shindo
        rows.append(feats)

        if (i + 1) % 1000 == 0:
            print(f" generated {i + 1}/{n_samples}")

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = build()
    print("\nClass distribution (this imbalance is realistic, not a bug):")
    print(df["shindo"].value_counts().reindex(SHINDO_LABELS, fill_value=0))
    out_path = "eew_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows -> {out_path}")