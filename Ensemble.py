import numpy as np
import pandas as pd

EPS = 1e-12

MAXIMIZE = {"ASW", "MoranI"}  # higher is better
MINIMIZE = {"PAS", "CHAOS", "GearyC"}  # lower is better


def minmax_norm(col: np.ndarray) -> np.ndarray:
    """Min-max normalize a 1D array to [0,1]. If constant, return all 0.5."""
    mn = np.min(col)
    mx = np.max(col)
    if abs(mx - mn) < EPS:
        return np.full_like(col, 0.5, dtype=float)
    return (col - mn) / (mx - mn + EPS)


def select_ensemble_method_per_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensemble selector (method-agnostic, no hardcoded thresholds).

    Input df columns (required):
      - sample : sample id (int/str)
      - method : method name (str)
      - ASW, PAS, CHAOS, MoranI, GearyC : floats
      - ARI : float (optional; used only for reporting if present)

    Output:
      Per-sample selected method + (optional) selected ARI.
    """
    required = {"sample", "method", "ASW", "PAS", "CHAOS", "MoranI", "GearyC"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out_rows = []

    # Process each sample independently
    for sample_id, g in df.groupby("sample", sort=False):
        g = g.copy().reset_index(drop=True)

        # Build normalized benefit-form matrix z_{m,k} in [0,1], higher is better
        Z = {}
        for k in (MAXIMIZE | MINIMIZE):
            z = minmax_norm(g[k].to_numpy(dtype=float))
            if k in MINIMIZE:
                z = 1.0 - z  # convert to "higher is better"
            Z[k] = z
        Z = pd.DataFrame(Z)

        # Reference (target) profile: max for ASW & MoranI; median for PAS/CHAOS/GearyC
        t = {}
        for k in Z.columns:
            if k in MAXIMIZE:
                t[k] = float(np.max(Z[k].to_numpy()))
            else:
                t[k] = float(np.median(Z[k].to_numpy()))
        t_vec = np.array([t[k] for k in Z.columns], dtype=float)

        # Choose method minimizing squared distance to target profile
        Z_mat = Z.to_numpy(dtype=float)
        d2 = np.sum((Z_mat - t_vec[None, :]) ** 2, axis=1)
        best_idx = int(np.argmin(d2))

        chosen = {
            "sample": sample_id,
            "ensemble_method": g.loc[best_idx, "method"],
        }
        if "ARI" in g.columns:
            chosen["ensemble_ARI"] = float(g.loc[best_idx, "ARI"])
        out_rows.append(chosen)

    return pd.DataFrame(out_rows)


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    # Example: build df from your data (fill with your rows)
    data = [
        # sample, method, ARI, ASW, PAS, CHAOS, MoranI, GearyC
        (151507, "STAGATE", 0.59, 0.04, 0.04, 0.06, 0.28, 0.72),
        (151507, "GraphST", 0.43, 0.01, 0.01, 0.06, 0.21, 0.79),
        (151507, "Scatter", 0.47131, -0.01766, 0.02227, 0.06, 0.213713, 0.786379),
        (151507, "ACT", 0.4862831344, 0.0254885392, 0.0203743189, 0.0563971327, 0.2096848416, 0.7906659313),
        (151507, "FACT", 0.5537604317, 0.0288000685, 0.0097133381, 0.0562859688, 0.2397395313, 0.7601504211),
        # ... add remaining samples similarly ...
    ]
    df = pd.DataFrame(data, columns=["sample", "method", "ARI", "ASW", "PAS", "CHAOS", "MoranI", "GearyC"])

    selected = select_ensemble_method_per_sample(df)
    print(selected)
