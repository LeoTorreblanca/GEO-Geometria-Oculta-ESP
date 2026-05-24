"""
GEO_CLASS — Internal Growth Fit

Purpose:
Compare LCDM vs GEO_CLASS using an internal growth-focused fit.

This script evaluates:
- fσ8 data
- S8 prior
- rd prior
- omega_b h² prior
- omega_m h² prior
- primordial amplitude prior

It is a compact internal validation test for the GEO_CLASS perturbative coupling
implemented in CLASS through the parameter `geo_xi`.

Outputs:
- CSV summary
- CSV prediction table
- PNG comparison plot
- execution log
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from classy import Class
from scipy.optimize import differential_evolution


BASE_DIR = Path(__file__).resolve().parent
CSV_DIR = BASE_DIR / "outputs" / "csv"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"
LOGS_DIR = BASE_DIR / "outputs" / "logs"

for folder in (CSV_DIR, PLOTS_DIR, LOGS_DIR):
    folder.mkdir(parents=True, exist_ok=True)


# =========================
# fσ8 data
# =========================

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

z_data = np.array([x[0] for x in FS8_DATA])
fs8_obs = np.array([x[1] for x in FS8_DATA])
fs8_err = np.array([x[2] for x in FS8_DATA])


# =========================
# Priors
# =========================

OMEGA_B = 0.05

S8_OBS, S8_ERR = 0.776, 0.0325
RD_OBS, RD_ERR = 147.1, 0.3
OMBH2_OBS, OMBH2_ERR = 0.0224, 0.0001
OMMH2_OBS, OMMH2_ERR = 0.143, 0.002

LOGA_OBS = 3.044
LOGA_ERR = 0.014


# =========================
# CLASS engine
# =========================

def run_class(h: float, omega_m: float, logA: float, geo_xi: float):
    A_s = np.exp(logA) / 1e10

    omega_b = OMEGA_B * h**2
    omega_cdm = (omega_m - OMEGA_B) * h**2

    if omega_cdm <= 0:
        raise ValueError("omega_cdm <= 0")

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
    sigma8_0 = cosmo.sigma8()
    D = cosmo.scale_independent_growth_factor(z)
    f = cosmo.scale_independent_growth_factor_f(z)
    return float(f * sigma8_0 * D)


def chi2_fs8(cosmo) -> float:
    pred = np.array([fs8_class(cosmo, z) for z in z_data])
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


# =========================
# Fit
# =========================

def fit_model(model: str) -> dict:
    if model == "LCDM":
        bounds = [
            (0.62, 0.75),   # h
            (0.22, 0.42),   # Omega_m
            (2.8, 3.3),     # log(1e10 A_s)
        ]
    elif model == "GEO_CLASS":
        bounds = [
            (0.62, 0.75),   # h
            (0.22, 0.42),   # Omega_m
            (2.8, 3.3),     # log(1e10 A_s)
            (0.0, 1.0),     # geo_xi
        ]
    else:
        raise ValueError(model)

    def obj(x):
        if model == "LCDM":
            h, omega_m, logA = x
            geo_xi = 1.0
        else:
            h, omega_m, logA, geo_xi = x

        if omega_m <= OMEGA_B:
            return 1e30

        try:
            cosmo, _ = run_class(h, omega_m, logA, geo_xi)
            chi_rd_val, _ = chi2_rd(cosmo)

            val = (
                chi2_fs8(cosmo)
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

    res = differential_evolution(
        obj,
        bounds,
        seed=123,
        polish=True,
        tol=1e-4,
        maxiter=45,
        popsize=8,
    )

    if model == "LCDM":
        h, omega_m, logA = res.x
        geo_xi = 1.0
    else:
        h, omega_m, logA, geo_xi = res.x

    cosmo, A_s = run_class(h, omega_m, logA, geo_xi)

    chi_fs8_val = chi2_fs8(cosmo)
    chi_s8_val = chi2_s8(cosmo, omega_m)
    chi_rd_val, rd = chi2_rd(cosmo)
    chi_ob_val = chi2_ombh2(h)
    chi_om_val = chi2_ommh2(h, omega_m)
    chi_As_val = chi2_As(logA)

    chi_total = (
        chi_fs8_val
        + chi_s8_val
        + chi_rd_val
        + chi_ob_val
        + chi_om_val
        + chi_As_val
    )

    sigma8 = float(cosmo.sigma8())
    S8 = S8_val(cosmo, omega_m)

    if model == "LCDM":
        mu_eff = 1.0
    else:
        mu_eff = (OMEGA_B + geo_xi * (omega_m - OMEGA_B)) / omega_m

    predictions = np.array([fs8_class(cosmo, z) for z in z_data])

    k = 3 if model == "LCDM" else 4
    n = len(FS8_DATA) + 5

    result = {
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
        "chi_fs8": float(chi_fs8_val),
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

    return result


def save_outputs(lcdm: dict, geo: dict) -> None:
    summary_rows = []
    for r in (lcdm, geo):
        row = {k: v for k, v in r.items() if k != "predictions"}
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = CSV_DIR / "geo_growth_internal_fit_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    pred_df = pd.DataFrame({
        "z": z_data,
        "fs8_obs": fs8_obs,
        "fs8_err": fs8_err,
        "fs8_LCDM": lcdm["predictions"],
        "fs8_GEO_CLASS": geo["predictions"],
        "residual_LCDM": fs8_obs - lcdm["predictions"],
        "residual_GEO_CLASS": fs8_obs - geo["predictions"],
    })
    pred_path = CSV_DIR / "geo_growth_internal_fit_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    plt.figure()
    plt.errorbar(z_data, fs8_obs, yerr=fs8_err, fmt="o", label="Observed fσ8")
    plt.plot(z_data, lcdm["predictions"], marker="o", label="LCDM")
    plt.plot(z_data, geo["predictions"], marker="o", label="GEO_CLASS")
    plt.xlabel("z")
    plt.ylabel("fσ8")
    plt.title("GEO_CLASS internal growth fit")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "geo_growth_internal_fit_fs8.png", dpi=160)
    plt.close()

    log_path = LOGS_DIR / "geo_growth_internal_fit.log"
    log_path.write_text(
        "GEO_CLASS internal growth fit completed.\n"
        f"Delta chi2 GEO-LCDM = {geo['chi_total'] - lcdm['chi_total']}\n"
        f"Delta AIC GEO-LCDM  = {geo['AIC'] - lcdm['AIC']}\n"
        f"Delta BIC GEO-LCDM  = {geo['BIC'] - lcdm['BIC']}\n"
        f"CSV summary: {summary_path}\n"
        f"CSV predictions: {pred_path}\n",
        encoding="utf-8",
    )


def print_result(r: dict) -> None:
    print("\nModel:", r["model"])
    for key in [
        "H0", "h", "Omega_m", "Omega_b", "Omega_d", "logA", "A_s",
        "sigma8", "S8", "rd", "geo_xi", "mu_eff", "Omega_growth",
        "chi_total", "chi_fs8", "chi_S8", "chi_rd", "chi_ombh2",
        "chi_ommh2", "chi_As", "AIC", "BIC",
    ]:
        print(f"{key:14s} = {r[key]}")


def main() -> None:
    print("\n==============================")
    print("GEO_CLASS — INTERNAL GROWTH FIT")
    print("==============================")

    lcdm = fit_model("LCDM")
    geo = fit_model("GEO_CLASS")

    for r in (lcdm, geo):
        print_result(r)

    print("\n==============================")
    print("COMPARISON")
    print("==============================")
    print("Delta chi2 GEO - LCDM =", geo["chi_total"] - lcdm["chi_total"])
    print("Delta AIC  GEO - LCDM =", geo["AIC"] - lcdm["AIC"])
    print("Delta BIC  GEO - LCDM =", geo["BIC"] - lcdm["BIC"])

    save_outputs(lcdm, geo)

    print("\nSaved outputs:")
    print(" - outputs/csv/geo_growth_internal_fit_summary.csv")
    print(" - outputs/csv/geo_growth_internal_fit_predictions.csv")
    print(" - outputs/plots/geo_growth_internal_fit_fs8.png")
    print(" - outputs/logs/geo_growth_internal_fit.log\n")


if __name__ == "__main__":
    main()
