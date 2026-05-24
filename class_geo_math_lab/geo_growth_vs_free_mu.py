"""
GEO_CLASS — GEO vs Free-Mu Comparison

Purpose:
Compare three growth scenarios:

- LCDM
- GEO_CLASS
- FREE_MU

Goal:
Determine whether the GEO_CLASS suppression behaves as a structured geometric
coupling or if it is indistinguishable from a generic free growth parameter μ.

Outputs:
- CSV summaries
- comparison plots
- execution log
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from classy import Class
from scipy.optimize import differential_evolution


# ==================================================
# Paths
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

CSV_DIR = BASE_DIR / "outputs" / "csv"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"
LOGS_DIR = BASE_DIR / "outputs" / "logs"

for folder in (CSV_DIR, PLOTS_DIR, LOGS_DIR):
    folder.mkdir(parents=True, exist_ok=True)


# ==================================================
# Data
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

z_data = np.array([x[0] for x in FS8_DATA])
fs8_obs = np.array([x[1] for x in FS8_DATA])
fs8_err = np.array([x[2] for x in FS8_DATA])


# ==================================================
# Priors
# ==================================================

OMEGA_B = 0.05

S8_OBS, S8_ERR = 0.776, 0.0325
RD_OBS, RD_ERR = 147.1, 0.3
OMBH2_OBS, OMBH2_ERR = 0.0224, 0.0001
OMMH2_OBS, OMMH2_ERR = 0.143, 0.002
LOGA_OBS, LOGA_ERR = 3.044, 0.014


# ==================================================
# CLASS engine
# ==================================================

def run_class(h, omega_m, logA, geo_xi, geo_mu, geo_mode):
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
        "geo_mu": geo_mu,
        "geo_mode": geo_mode,
    })
    cosmo.compute()

    return cosmo, A_s


def fs8_class(cosmo, z):
    return (
        cosmo.scale_independent_growth_factor_f(z)
        * cosmo.sigma8()
        * cosmo.scale_independent_growth_factor(z)
    )


def chi2_fs8(cosmo):
    pred = np.array([fs8_class(cosmo, z) for z in z_data])
    return float(np.sum(((fs8_obs - pred) / fs8_err) ** 2))


def S8_val(cosmo, omega_m):
    return float(cosmo.sigma8() * np.sqrt(omega_m / 0.3))


def chi2_s8(cosmo, omega_m):
    return float(((S8_val(cosmo, omega_m) - S8_OBS) / S8_ERR) ** 2)


def chi2_rd(cosmo):
    try:
        rd = float(cosmo.rs_drag())
    except Exception:
        rd = RD_OBS

    return float(((rd - RD_OBS) / RD_ERR) ** 2), rd


def chi2_ombh2(h):
    return float(((OMEGA_B * h**2 - OMBH2_OBS) / OMBH2_ERR) ** 2)


def chi2_ommh2(h, omega_m):
    return float(((omega_m * h**2 - OMMH2_OBS) / OMMH2_ERR) ** 2)


def chi2_As(logA):
    return float(((logA - LOGA_OBS) / LOGA_ERR) ** 2)


# ==================================================
# GEO equivalence
# ==================================================

def xi_equiv_from_mu(mu_eff, omega_m):
    od = omega_m - OMEGA_B

    if od <= 0:
        return np.nan

    return (mu_eff * omega_m - OMEGA_B) / od


# ==================================================
# Fit
# ==================================================

def fit_model(model):

    if model == "LCDM":
        bounds = [(0.62, 0.75), (0.22, 0.42), (2.8, 3.3)]
        k = 3

    elif model == "GEO_CLASS":
        bounds = [(0.62, 0.75), (0.22, 0.42), (2.8, 3.3), (0.0, 1.0)]
        k = 4

    elif model == "FREE_MU":
        bounds = [(0.62, 0.75), (0.22, 0.42), (2.8, 3.3), (0.0, 1.5)]
        k = 4

    else:
        raise ValueError(model)

    def unpack(x):

        if model == "LCDM":
            h, omega_m, logA = x
            geo_xi = 1.0
            geo_mu = 1.0
            geo_mode = 0

        elif model == "GEO_CLASS":
            h, omega_m, logA, geo_xi = x
            geo_mu = 1.0
            geo_mode = 0

        else:
            h, omega_m, logA, geo_mu = x
            geo_xi = 1.0
            geo_mode = 1

        return h, omega_m, logA, geo_xi, geo_mu, geo_mode

    def objective(x):

        h, omega_m, logA, geo_xi, geo_mu, geo_mode = unpack(x)

        if omega_m <= OMEGA_B:
            return 1e30

        try:

            cosmo, _ = run_class(
                h,
                omega_m,
                logA,
                geo_xi,
                geo_mu,
                geo_mode,
            )

            chi_rd_val, _ = chi2_rd(cosmo)

            total = (
                chi2_fs8(cosmo)
                + chi2_s8(cosmo, omega_m)
                + chi_rd_val
                + chi2_ombh2(h)
                + chi2_ommh2(h, omega_m)
                + chi2_As(logA)
            )

            cosmo.struct_cleanup()
            cosmo.empty()

            return float(total)

        except Exception:
            return 1e30

    res = differential_evolution(
        objective,
        bounds,
        seed=123,
        polish=True,
        tol=1e-4,
        maxiter=45,
        popsize=8,
    )

    h, omega_m, logA, geo_xi, geo_mu, geo_mode = unpack(res.x)

    cosmo, A_s = run_class(
        h,
        omega_m,
        logA,
        geo_xi,
        geo_mu,
        geo_mode,
    )

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
    s8 = S8_val(cosmo, omega_m)

    if model == "LCDM":
        mu_eff = 1.0
        xi_equiv = 1.0

    elif model == "GEO_CLASS":
        mu_eff = (OMEGA_B + geo_xi * (omega_m - OMEGA_B)) / omega_m
        xi_equiv = geo_xi

    else:
        mu_eff = geo_mu
        xi_equiv = xi_equiv_from_mu(mu_eff, omega_m)

    n = len(FS8_DATA) + 5

    result = {
        "model": model,
        "H0": float(100 * h),
        "Omega_m": float(omega_m),
        "Omega_b": float(OMEGA_B),
        "Omega_d": float(omega_m - OMEGA_B),
        "logA": float(logA),
        "A_s": float(A_s),
        "sigma8": sigma8,
        "S8": s8,
        "rd": float(rd),
        "geo_xi": float(geo_xi),
        "geo_mu": float(geo_mu),
        "mu_eff": float(mu_eff),
        "xi_equiv": float(xi_equiv),
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
    }

    cosmo.struct_cleanup()
    cosmo.empty()

    return result


# ==================================================
# Outputs
# ==================================================

def save_outputs(results):

    df = pd.DataFrame(results)

    csv_path = CSV_DIR / "geo_growth_vs_free_mu_summary.csv"
    df.to_csv(csv_path, index=False)

    plt.figure()
    plt.bar(df["model"], df["chi_total"])
    plt.ylabel("chi2 total")
    plt.title("GEO_CLASS vs FREE_MU")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "geo_growth_vs_free_mu_chi2.png", dpi=160)
    plt.close()

    plt.figure()
    plt.bar(df["model"], df["AIC"])
    plt.ylabel("AIC")
    plt.title("GEO_CLASS vs FREE_MU")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "geo_growth_vs_free_mu_AIC.png", dpi=160)
    plt.close()

    plt.figure()
    plt.bar(df["model"], df["BIC"])
    plt.ylabel("BIC")
    plt.title("GEO_CLASS vs FREE_MU")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "geo_growth_vs_free_mu_BIC.png", dpi=160)
    plt.close()

    log_path = LOGS_DIR / "geo_growth_vs_free_mu.log"

    lcdm = results[0]
    geo = results[1]
    mu = results[2]

    log_path.write_text(
        "GEO_CLASS vs FREE_MU comparison completed.\n"
        f"Delta chi2 GEO-LCDM = {geo['chi_total'] - lcdm['chi_total']}\n"
        f"Delta chi2 MU-GEO   = {mu['chi_total'] - geo['chi_total']}\n"
        f"MU xi_equiv         = {mu['xi_equiv']}\n",
        encoding="utf-8",
    )


# ==================================================
# Main
# ==================================================

def print_result(r):

    print("\nModel:", r["model"])

    for key in [
        "H0",
        "Omega_m",
        "Omega_b",
        "Omega_d",
        "logA",
        "A_s",
        "sigma8",
        "S8",
        "rd",
        "geo_xi",
        "geo_mu",
        "mu_eff",
        "xi_equiv",
        "Omega_growth",
        "chi_total",
        "chi_fs8",
        "chi_S8",
        "chi_rd",
        "chi_ombh2",
        "chi_ommh2",
        "chi_As",
        "AIC",
        "BIC",
    ]:
        print(f"{key:14s} = {r[key]}")


def main():

    print("\n==============================")
    print("GEO_CLASS vs FREE_MU")
    print("==============================")

    results = [
        fit_model("LCDM"),
        fit_model("GEO_CLASS"),
        fit_model("FREE_MU"),
    ]

    for r in results:
        print_result(r)

    lcdm = results[0]
    geo = results[1]
    mu = results[2]

    print("\n==============================")
    print("COMPARISON")
    print("==============================")

    for r in results:
        print("\nModel:", r["model"])
        print("Delta chi2 vs LCDM =", r["chi_total"] - lcdm["chi_total"])
        print("Delta AIC  vs LCDM =", r["AIC"] - lcdm["AIC"])
        print("Delta BIC  vs LCDM =", r["BIC"] - lcdm["BIC"])

    print("\n==============================")
    print("GEO_CLASS vs FREE_MU")
    print("==============================")
    print("Delta chi2 MU - GEO =", mu["chi_total"] - geo["chi_total"])
    print("Delta AIC  MU - GEO =", mu["AIC"] - geo["AIC"])
    print("Delta BIC  MU - GEO =", mu["BIC"] - geo["BIC"])
    print("MU xi_equiv         =", mu["xi_equiv"])
    print("GEO xi              =", geo["geo_xi"])
    print("MU physical GEO?    =", 0.0 <= mu["xi_equiv"] <= 1.0)

    print("\nInterpretation:")
    print("- If FREE_MU strongly beats GEO_CLASS in BIC, GEO_CLASS is incomplete.")
    print("- If both are comparable, FREE_MU does not dethrone GEO_CLASS.")
    print("- If xi_equiv is between 0 and 1, FREE_MU can be reinterpreted geometrically.")

    save_outputs(results)

    print("\nSaved outputs:")
    print(" - outputs/csv/geo_growth_vs_free_mu_summary.csv")
    print(" - outputs/plots/geo_growth_vs_free_mu_chi2.png")
    print(" - outputs/plots/geo_growth_vs_free_mu_AIC.png")
    print(" - outputs/plots/geo_growth_vs_free_mu_BIC.png")
    print(" - outputs/logs/geo_growth_vs_free_mu.log\n")


if __name__ == "__main__":
    main()
