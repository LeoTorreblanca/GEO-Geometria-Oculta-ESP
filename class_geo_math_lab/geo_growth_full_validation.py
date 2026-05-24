"""
GEO_CLASS — Full Cosmological Validation

Purpose:
Compare LCDM vs GEO_CLASS with a broader cosmological likelihood set:

- Pantheon+SH0ES supernovae
- DESI BAO files
- fσ8 growth data
- S8 prior
- sound horizon rd prior
- omega_b h² prior
- omega_m h² prior
- primordial amplitude prior

This script uses local relative data paths and writes reproducible outputs:
CSV summaries, prediction tables, plots and logs.
"""

from pathlib import Path
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from classy import Class
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import differential_evolution


# ==================================================
# Paths
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
PANTHEON_DIR = DATA_DIR / "pantheon"
BAO_ROOT = DATA_DIR / "bao"

SN_FILE = PANTHEON_DIR / "Pantheon+SH0ES.dat"
SN_COV_FILE = PANTHEON_DIR / "Pantheon+SH0ES_STAT+SYS.cov"

CSV_DIR = BASE_DIR / "outputs" / "csv"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"
LOGS_DIR = BASE_DIR / "outputs" / "logs"

for folder in (CSV_DIR, PLOTS_DIR, LOGS_DIR):
    folder.mkdir(parents=True, exist_ok=True)


# ==================================================
# Constants / priors
# ==================================================

OMEGA_B = 0.05

S8_OBS, S8_ERR = 0.776, 0.0325
RD_OBS, RD_ERR = 147.1, 0.3
OMBH2_OBS, OMBH2_ERR = 0.0224, 0.0001
OMMH2_OBS, OMMH2_ERR = 0.143, 0.002
LOGA_OBS, LOGA_ERR = 3.044, 0.014


# ==================================================
# fσ8 data
# ==================================================

FS8_DATA = [
    (0.02, 0.360, 0.040), (0.067, 0.423, 0.055),
    (0.10, 0.370, 0.130), (0.15, 0.490, 0.050),
    (0.17, 0.510, 0.060), (0.22, 0.420, 0.070),
    (0.25, 0.351, 0.058), (0.32, 0.384, 0.095),
    (0.37, 0.460, 0.038), (0.38, 0.430, 0.054),
    (0.44, 0.413, 0.080), (0.51, 0.452, 0.057),
    (0.57, 0.444, 0.038), (0.60, 0.390, 0.063),
    (0.61, 0.457, 0.052), (0.73, 0.437, 0.072),
    (0.80, 0.470, 0.080), (0.86, 0.400, 0.110),
    (1.40, 0.482, 0.116), (1.52, 0.426, 0.077),
    (1.944, 0.364, 0.106),
]

z_fs8 = np.array([x[0] for x in FS8_DATA])
fs8_obs = np.array([x[1] for x in FS8_DATA])
fs8_err = np.array([x[2] for x in FS8_DATA])


# ==================================================
# Pantheon
# ==================================================

def load_pantheon():
    if not SN_FILE.exists():
        raise FileNotFoundError(f"Missing Pantheon file: {SN_FILE}")
    if not SN_COV_FILE.exists():
        raise FileNotFoundError(f"Missing Pantheon covariance file: {SN_COV_FILE}")

    data = np.genfromtxt(SN_FILE, names=True, dtype=None, encoding=None)
    names = data.dtype.names

    z_col = "zHD" if "zHD" in names else ("zcmb" if "zcmb" in names else names[1])

    if "m_b_corr" in names:
        mag_col = "m_b_corr"
    elif "MU_SH0ES" in names:
        mag_col = "MU_SH0ES"
    elif "MU" in names:
        mag_col = "MU"
    else:
        raise ValueError("Could not find magnitude/MU column in Pantheon data.")

    z = np.array(data[z_col], dtype=float)
    m = np.array(data[mag_col], dtype=float)

    raw = np.loadtxt(SN_COV_FILE)
    if raw.ndim == 1:
        if int(raw[0]) == len(z):
            cov = raw[1:].reshape(len(z), len(z))
        else:
            cov = raw.reshape(len(z), len(z))
    else:
        cov = raw

    cho = cho_factor(cov, lower=True, check_finite=False)
    ones = np.ones(len(z))

    print("Pantheon loaded:", len(z), "supernovae")
    return z, m, cho, ones


z_sn, m_sn, cov_sn_cho, ones_sn = load_pantheon()


def chi2_sn(cosmo) -> float:
    dl = np.array([cosmo.luminosity_distance(float(z)) for z in z_sn])
    mu_th = 5.0 * np.log10(dl) + 25.0

    diff = m_sn - mu_th

    c_inv_d = cho_solve(cov_sn_cho, diff, check_finite=False)
    c_inv_1 = cho_solve(cov_sn_cho, ones_sn, check_finite=False)

    a = float(diff @ c_inv_d)
    b = float(diff @ c_inv_1)
    cval = float(ones_sn @ c_inv_1)

    return float(a - b * b / cval)


# ==================================================
# BAO / DESI
# ==================================================

def read_numbers(path: str):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip().startswith("#"):
                continue
            nums = re.findall(
                r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?",
                line,
            )
            if nums:
                rows.append([float(x) for x in nums])

    if not rows:
        raise ValueError("No numerical data found.")

    lens = [len(r) for r in rows]
    if len(set(lens)) == 1:
        arr = np.array(rows)
        if arr.shape[1] == 1:
            return arr[:, 0]
        return arr

    return np.array([x for r in rows for x in r])


def infer_z(path: str, mean_raw=None):
    name = os.path.basename(path).lower()
    arr = np.asarray(mean_raw) if mean_raw is not None else None

    if arr is not None and arr.ndim == 1 and len(arr) == 2:
        if 0.01 < arr[0] < 4:
            return float(arr[0])

    if arr is not None and arr.ndim == 2 and arr.shape[1] >= 2:
        z_col = arr[:, 0]
        if np.all((z_col > 0.01) & (z_col < 4)):
            return float(np.mean(z_col))

    m = re.search(r"z([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)", name)
    if m:
        return 0.5 * (float(m.group(1)) + float(m.group(2)))

    if "bgs" in name:
        return 0.295
    if "lrg+elg" in name:
        return 0.934
    if "elg" in name:
        return 1.321
    if "qso" in name:
        return 1.484
    if "lya" in name:
        return 2.33

    return None


def parse_bao_pair(mean_path: str, cov_path: str):
    mean_raw = read_numbers(mean_path)
    cov_raw = read_numbers(cov_path)

    name = os.path.basename(mean_path).lower()
    z = infer_z(mean_path, mean_raw)

    mean_arr = np.asarray(mean_raw)
    cov_arr = np.asarray(cov_raw)

    if mean_arr.ndim == 1 and len(mean_arr) == 2 and cov_arr.ndim == 1 and len(cov_arr) == 1:
        return {
            "z": z,
            "obs": ["DV"],
            "mean": np.array([mean_arr[1]]),
            "cov": np.array([[cov_arr[0]]]),
            "name": os.path.basename(mean_path),
        }

    if mean_arr.ndim == 2 and mean_arr.shape[1] >= 2:
        mean_vec = mean_arr[:, -1]
    else:
        mean_vec = mean_arr.flatten()

    if cov_arr.ndim == 1:
        if len(cov_arr) == len(mean_vec):
            cov = np.diag(cov_arr)
        elif len(cov_arr) == len(mean_vec) ** 2:
            cov = cov_arr.reshape(len(mean_vec), len(mean_vec))
        else:
            raise ValueError("Incompatible covariance shape.")
    else:
        cov = cov_arr

    if "lya" in name:
        obs = ["DH", "DM"]
    elif len(mean_vec) == 2:
        obs = ["DM", "DH"]
    elif len(mean_vec) == 1:
        obs = ["DV"]
    else:
        raise ValueError("Could not interpret BAO observable.")

    return {
        "z": z,
        "obs": obs,
        "mean": mean_vec,
        "cov": cov,
        "name": os.path.basename(mean_path),
    }


def load_bao(token: str):
    files = glob.glob(str(BAO_ROOT / "**" / "*"), recursive=True)
    files = [f for f in files if os.path.isfile(f)]

    means = [
        f for f in files
        if f.lower().endswith("_mean.txt") or f.lower().endswith("_mean")
    ]
    covs = [
        f for f in files
        if f.lower().endswith("_cov.txt") or f.lower().endswith("_cov")
    ]

    entries = []

    for mf in means:
        low = mf.lower()
        if token not in low:
            continue
        if "all_gccomb" in low:
            continue

        base_m = re.sub(r"_mean(\.txt)?$", "", mf, flags=re.I)

        for cf in covs:
            base_c = re.sub(r"_cov(\.txt)?$", "", cf, flags=re.I)
            if os.path.normcase(base_m) == os.path.normcase(base_c):
                try:
                    entries.append(parse_bao_pair(mf, cf))
                except Exception as e:
                    print("SKIP BAO:", os.path.basename(mf), e)
                break

    return entries


BAO = load_bao("desi_gaussian_bao")
print("BAO points:", sum(len(e["mean"]) for e in BAO))


def bao_predict(cosmo, z: float, obs: str, rd: float) -> float:
    dm = (1.0 + z) * cosmo.angular_distance(z)
    dh = 1.0 / cosmo.Hubble(z)
    dv = (z * dm * dm * dh) ** (1.0 / 3.0)

    if obs == "DM":
        return float(dm / rd)
    if obs == "DH":
        return float(dh / rd)
    if obs == "DV":
        return float(dv / rd)

    raise ValueError(obs)


def chi2_bao(cosmo) -> float:
    try:
        rd = float(cosmo.rs_drag())
    except Exception:
        rd = RD_OBS

    total = 0.0
    for entry in BAO:
        pred = np.array([bao_predict(cosmo, entry["z"], ob, rd) for ob in entry["obs"]])
        diff = entry["mean"] - pred
        inv = np.linalg.pinv(entry["cov"])
        total += float(diff.T @ inv @ diff)

    return float(total)


# ==================================================
# CLASS + likelihoods
# ==================================================

def run_class(h: float, omega_m: float, logA: float, geo_xi: float, model: str):
    A_s = np.exp(logA) / 1e10
    omega_b = OMEGA_B * h**2
    omega_cdm = (omega_m - OMEGA_B) * h**2

    if omega_cdm <= 0:
        raise ValueError("omega_cdm <= 0")

    if model == "LCDM":
        geo_xi = 1.0

    cosmo = Class()
    cosmo.set({
        "h": h,
        "omega_b": omega_b,
        "omega_cdm": omega_cdm,
        "A_s": A_s,
        "n_s": 0.965,
        "tau_reio": 0.054,
        "output": "mPk",
        "P_k_max_1/Mpc": 2.0,
        "z_max_pk": 2.5,
        "geo_xi": geo_xi,
        "geo_mu": 1.0,
        "geo_mode": 0,
    })
    cosmo.compute()

    return cosmo, A_s


def fs8_class(cosmo, z: float) -> float:
    return float(
        cosmo.scale_independent_growth_factor_f(z)
        * cosmo.sigma8()
        * cosmo.scale_independent_growth_factor(z)
    )


def chi2_fs8(cosmo) -> float:
    pred = np.array([fs8_class(cosmo, z) for z in z_fs8])
    return float(np.sum(((fs8_obs - pred) / fs8_err) ** 2))


def S8_val(cosmo, omega_m: float) -> float:
    return float(cosmo.sigma8() * np.sqrt(omega_m / 0.3))


def chi2_s8(cosmo, omega_m: float) -> float:
    return float(((S8_val(cosmo, omega_m) - S8_OBS) / S8_ERR) ** 2)


def chi2_rd(cosmo):
    try:
        rd = float(cosmo.rs_drag())
    except Exception:
        rd = RD_OBS
    return float(((rd - RD_OBS) / RD_ERR) ** 2), rd


def chi2_ombh2(h: float) -> float:
    return float(((OMEGA_B * h**2 - OMBH2_OBS) / OMBH2_ERR) ** 2)


def chi2_ommh2(h: float, omega_m: float) -> float:
    return float(((omega_m * h**2 - OMMH2_OBS) / OMMH2_ERR) ** 2)


def chi2_As(logA: float) -> float:
    return float(((logA - LOGA_OBS) / LOGA_ERR) ** 2)


# ==================================================
# Fit
# ==================================================

def fit_model(model: str) -> dict:
    if model == "LCDM":
        bounds = [
            (0.62, 0.75),  # h
            (0.22, 0.42),  # Omega_m
            (2.8, 3.3),    # log(1e10 A_s)
        ]
        k = 3
    elif model == "GEO_CLASS":
        bounds = [
            (0.62, 0.75),  # h
            (0.22, 0.42),  # Omega_m
            (2.8, 3.3),    # log(1e10 A_s)
            (0.0, 1.0),    # geo_xi
        ]
        k = 4
    else:
        raise ValueError(model)

    def unpack(x):
        if model == "LCDM":
            h, omega_m, logA = x
            geo_xi = 1.0
        else:
            h, omega_m, logA, geo_xi = x
        return h, omega_m, logA, geo_xi

    def obj(x):
        h, omega_m, logA, geo_xi = unpack(x)

        if omega_m <= OMEGA_B:
            return 1e30

        try:
            cosmo, _ = run_class(h, omega_m, logA, geo_xi, model)
            chi_rd_val, _ = chi2_rd(cosmo)

            val = (
                chi2_bao(cosmo)
                + chi2_fs8(cosmo)
                + chi2_sn(cosmo)
                + chi2_s8(cosmo, omega_m)
                + chi_rd_val
                + chi2_ombh2(h)
                + chi2_ommh2(h, omega_m)
                + chi2_As(logA)
            )

            cosmo.struct_cleanup()
            cosmo.empty()
            return float(val)

        except Exception:
            return 1e30

    result = differential_evolution(
        obj,
        bounds,
        seed=123,
        polish=True,
        tol=1e-4,
        maxiter=45,
        popsize=8,
    )

    h, omega_m, logA, geo_xi = unpack(result.x)
    cosmo, A_s = run_class(h, omega_m, logA, geo_xi, model)

    chi_fs8_val = chi2_fs8(cosmo)
    chi_sn_val = chi2_sn(cosmo)
    chi_bao_val = chi2_bao(cosmo)
    chi_s8_val = chi2_s8(cosmo, omega_m)
    chi_rd_val, rd = chi2_rd(cosmo)
    chi_ob_val = chi2_ombh2(h)
    chi_om_val = chi2_ommh2(h, omega_m)
    chi_As_val = chi2_As(logA)

    chi_total = (
        chi_bao_val
        + chi_fs8_val
        + chi_sn_val
        + chi_s8_val
        + chi_rd_val
        + chi_ob_val
        + chi_om_val
        + chi_As_val
    )

    sigma8 = float(cosmo.sigma8())
    S8 = S8_val(cosmo, omega_m)
    mu_eff = 1.0 if model == "LCDM" else (OMEGA_B + geo_xi * (omega_m - OMEGA_B)) / omega_m
    predictions = np.array([fs8_class(cosmo, z) for z in z_fs8])

    n = len(FS8_DATA) + len(z_sn) + sum(len(e["mean"]) for e in BAO) + 5

    output = {
        "model": model,
        "h": float(h),
        "H0": float(100 * h),
        "Omega_m": float(omega_m),
        "Omega_b": float(OMEGA_B),
        "Omega_d": float(omega_m - OMEGA_B),
        "logA": float(logA),
        "A_s": float(A_s),
        "sigma8": sigma8,
        "S8": S8,
        "rd": float(rd),
        "geo_xi": float(geo_xi),
        "mu_eff": float(mu_eff),
        "Omega_growth": float(mu_eff * omega_m),
        "chi_total": float(chi_total),
        "chi_bao": float(chi_bao_val),
        "chi_fs8": float(chi_fs8_val),
        "chi_sn": float(chi_sn_val),
        "chi_S8": float(chi_s8_val),
        "chi_rd": float(chi_rd_val),
        "chi_ombh2": float(chi_ob_val),
        "chi_ommh2": float(chi_om_val),
        "chi_As": float(chi_As_val),
        "AIC": float(chi_total + 2 * k),
        "BIC": float(chi_total + k * np.log(n)),
        "predictions": predictions,
    }

    cosmo.struct_cleanup()
    cosmo.empty()

    return output


def save_outputs(lcdm: dict, geo: dict) -> None:
    summary_rows = []
    for result in (lcdm, geo):
        row = {k: v for k, v in result.items() if k != "predictions"}
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = CSV_DIR / "geo_growth_full_validation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    pred_df = pd.DataFrame({
        "z": z_fs8,
        "fs8_obs": fs8_obs,
        "fs8_err": fs8_err,
        "fs8_LCDM": lcdm["predictions"],
        "fs8_GEO_CLASS": geo["predictions"],
        "residual_LCDM": fs8_obs - lcdm["predictions"],
        "residual_GEO_CLASS": fs8_obs - geo["predictions"],
    })
    pred_path = CSV_DIR / "geo_growth_full_validation_fs8_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    plt.figure()
    plt.errorbar(z_fs8, fs8_obs, yerr=fs8_err, fmt="o", label="Observed fσ8")
    plt.plot(z_fs8, lcdm["predictions"], marker="o", label="LCDM")
    plt.plot(z_fs8, geo["predictions"], marker="o", label="GEO_CLASS")
    plt.xlabel("z")
    plt.ylabel("fσ8")
    plt.title("GEO_CLASS full validation — fσ8")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "geo_growth_full_validation_fs8.png", dpi=160)
    plt.close()

    plt.figure()
    plt.axhline(0.0, linestyle="--")
    plt.errorbar(z_fs8, pred_df["residual_LCDM"], yerr=fs8_err, fmt="o", label="LCDM residual")
    plt.errorbar(z_fs8, pred_df["residual_GEO_CLASS"], yerr=fs8_err, fmt="o", label="GEO_CLASS residual")
    plt.xlabel("z")
    plt.ylabel("Observed - predicted")
    plt.title("GEO_CLASS full validation — fσ8 residuals")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "geo_growth_full_validation_fs8_residuals.png", dpi=160)
    plt.close()

    log_path = LOGS_DIR / "geo_growth_full_validation.log"
    log_path.write_text(
        "GEO_CLASS full cosmological validation completed.\n"
        f"Pantheon supernovae: {len(z_sn)}\n"
        f"BAO points: {sum(len(e['mean']) for e in BAO)}\n"
        f"Delta chi2 GEO-LCDM = {geo['chi_total'] - lcdm['chi_total']}\n"
        f"Delta AIC GEO-LCDM  = {geo['AIC'] - lcdm['AIC']}\n"
        f"Delta BIC GEO-LCDM  = {geo['BIC'] - lcdm['BIC']}\n"
        f"Summary CSV: {summary_path}\n"
        f"fσ8 prediction CSV: {pred_path}\n",
        encoding="utf-8",
    )


def print_result(result: dict) -> None:
    print("\nModel:", result["model"])
    for key in [
        "H0", "h", "Omega_m", "Omega_b", "Omega_d", "logA", "A_s",
        "sigma8", "S8", "rd", "geo_xi", "mu_eff", "Omega_growth",
        "chi_total", "chi_bao", "chi_fs8", "chi_sn", "chi_S8",
        "chi_rd", "chi_ombh2", "chi_ommh2", "chi_As", "AIC", "BIC",
    ]:
        print(f"{key:14s} = {result[key]}")


def main() -> None:
    print("\n==============================")
    print("GEO_CLASS — FULL COSMOLOGICAL VALIDATION")
    print("==============================")

    lcdm = fit_model("LCDM")
    geo = fit_model("GEO_CLASS")

    print_result(lcdm)
    print_result(geo)

    print("\n==============================")
    print("COMPARISON")
    print("==============================")
    print("Delta chi2 GEO - LCDM =", geo["chi_total"] - lcdm["chi_total"])
    print("Delta AIC  GEO - LCDM =", geo["AIC"] - lcdm["AIC"])
    print("Delta BIC  GEO - LCDM =", geo["BIC"] - lcdm["BIC"])

    save_outputs(lcdm, geo)

    print("\nSaved outputs:")
    print(" - outputs/csv/geo_growth_full_validation_summary.csv")
    print(" - outputs/csv/geo_growth_full_validation_fs8_predictions.csv")
    print(" - outputs/plots/geo_growth_full_validation_fs8.png")
    print(" - outputs/plots/geo_growth_full_validation_fs8_residuals.png")
    print(" - outputs/logs/geo_growth_full_validation.log\n")


if __name__ == "__main__":
    main()
