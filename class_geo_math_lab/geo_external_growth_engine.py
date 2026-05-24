"""
GEO_CLASS — External Growth Engine

Purpose:
Test GEO growth as an external effective operator on top of a standard CLASS
background.

Concept:
- CLASS provides the LCDM background H(z).
- GEO_CLASS modifies only the effective growth source:
    LCDM      : Omega_m(z)
    GEO_CLASS : Omega_b(z) + geo_xi * Omega_d(z)

This checks whether the GEO growth signal can be reproduced without relying on
the internal perturbation patch alone.

Outputs:
- CSV summary
- CSV predictions
- growth/residual plots
- execution log
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from classy import Class
from scipy.integrate import solve_ivp
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

z_data = np.array([x[0] for x in FS8_DATA])
fs8_obs = np.array([x[1] for x in FS8_DATA])
fs8_err = np.array([x[2] for x in FS8_DATA])

OMEGA_B = 0.05


# ==================================================
# CLASS background
# ==================================================

def run_class_background(h=0.6714, omega_m=0.32194):
    """
    Run standard CLASS background.

    The GEO operator is not injected here.
    CLASS provides H(z) and the baseline cosmological background.
    """

    omega_b = OMEGA_B * h**2
    omega_cdm = (omega_m - OMEGA_B) * h**2

    cosmo = Class()
    cosmo.set({
        "h": h,
        "omega_b": omega_b,
        "omega_cdm": omega_cdm,
        "A_s": 2.1e-9,
        "n_s": 0.965,
        "tau_reio": 0.054,
        "output": "mPk",
        "P_k_max_1/Mpc": 2.0,
        "z_max_pk": 2.5,
    })
    cosmo.compute()

    return cosmo, float(cosmo.sigma8())


def H_from_class(cosmo, z):
    return float(cosmo.Hubble(float(z)))


def dlnH_dz_class(cosmo, z):
    dz = 1e-4
    zp = z + dz
    zm = max(z - dz, 0.0)

    hp = H_from_class(cosmo, zp)
    hm = H_from_class(cosmo, zm)

    return float((np.log(hp) - np.log(hm)) / (zp - zm))


# ==================================================
# Effective densities
# ==================================================

def E2_LCDM(z, omega_m):
    return omega_m * (1 + z) ** 3 + (1 - omega_m)


def Omega_m_z(z, omega_m):
    return omega_m * (1 + z) ** 3 / E2_LCDM(z, omega_m)


def Omega_b_z(z, omega_b, omega_m):
    return omega_b * (1 + z) ** 3 / E2_LCDM(z, omega_m)


def Omega_d_z(z, omega_b, omega_m):
    return (omega_m - omega_b) * (1 + z) ** 3 / E2_LCDM(z, omega_m)


# ==================================================
# External GEO growth operator
# ==================================================

def growth_source(z, omega_m, omega_b, geo_xi, model):
    if model == "LCDM":
        return Omega_m_z(z, omega_m)

    if model == "GEO_CLASS":
        return Omega_b_z(z, omega_b, omega_m) + geo_xi * Omega_d_z(z, omega_b, omega_m)

    raise ValueError(model)


def solve_growth_external(cosmo, omega_m, omega_b, geo_xi, sigma8_0, model):
    zmax = 2.2
    z_eval = np.linspace(zmax, 0.0, 1200)

    def eq(z, y):
        delta, ddelta = y

        A = 2.0 / (1.0 + z) - dlnH_dz_class(cosmo, z)
        source = growth_source(z, omega_m, omega_b, geo_xi, model)
        B = 1.5 * source / (1.0 + z) ** 2

        return [
            ddelta,
            -A * ddelta + B * delta,
        ]

    y0 = [1e-5, -1e-5 / (1 + zmax)]

    sol = solve_ivp(
        eq,
        [zmax, 0.0],
        y0,
        t_eval=z_eval,
        rtol=1e-7,
        atol=1e-10,
    )

    if not sol.success:
        raise RuntimeError("solve_ivp failed.")

    z = sol.t
    delta = np.abs(sol.y[0])
    delta[delta < 1e-20] = 1e-20

    a = 1.0 / (1.0 + z)
    f = np.gradient(np.log(delta), np.log(a))

    delta0 = delta[-1]
    sigma8_z = sigma8_0 * delta / delta0
    fs8 = f * sigma8_z

    return z, fs8, delta, f


def predict_fs8(cosmo, omega_m, omega_b, geo_xi, sigma8_0, model, z_points):
    z_model, fs8_model, delta, f = solve_growth_external(
        cosmo,
        omega_m,
        omega_b,
        geo_xi,
        sigma8_0,
        model,
    )

    return (
        np.interp(z_points, z_model[::-1], fs8_model[::-1]),
        z_model,
        fs8_model,
        delta,
        f,
    )


def chi2_fs8(cosmo, omega_m, omega_b, geo_xi, sigma8_0, model):
    pred, *_ = predict_fs8(
        cosmo,
        omega_m,
        omega_b,
        geo_xi,
        sigma8_0,
        model,
        z_data,
    )

    return float(np.sum(((fs8_obs - pred) / fs8_err) ** 2))


# ==================================================
# Fit
# ==================================================

def fit_model(model):
    """
    Fit sigma8 and optionally geo_xi.

    LCDM:
        sigma8 is fitted.
    GEO_CLASS:
        sigma8 and geo_xi are fitted.
    """

    h = 0.6714
    omega_m = 0.32194
    omega_b = OMEGA_B

    cosmo, sigma8_class_raw = run_class_background(h=h, omega_m=omega_m)

    if model == "LCDM":
        bounds = [(0.45, 1.1)]

        def objective(x):
            sigma8_0 = x[0]
            return chi2_fs8(cosmo, omega_m, omega_b, 1.0, sigma8_0, "LCDM")

        result = differential_evolution(objective, bounds, seed=123, polish=True)
        sigma8_0 = float(result.x[0])
        geo_xi = 1.0

    elif model == "GEO_CLASS":
        bounds = [
            (0.45, 1.2),
            (0.0, 1.0),
        ]

        def objective(x):
            sigma8_0, geo_xi = x
            return chi2_fs8(cosmo, omega_m, omega_b, geo_xi, sigma8_0, "GEO_CLASS")

        result = differential_evolution(objective, bounds, seed=123, polish=True)
        sigma8_0 = float(result.x[0])
        geo_xi = float(result.x[1])

    else:
        raise ValueError(model)

    chi2 = chi2_fs8(cosmo, omega_m, omega_b, geo_xi, sigma8_0, model)

    mu_eff = (
        omega_b + geo_xi * (omega_m - omega_b)
    ) / omega_m if model == "GEO_CLASS" else 1.0

    predictions, z_curve, fs8_curve, delta_curve, f_curve = predict_fs8(
        cosmo,
        omega_m,
        omega_b,
        geo_xi,
        sigma8_0,
        model,
        z_data,
    )

    output = {
        "model": model,
        "h": float(h),
        "Omega_m": float(omega_m),
        "Omega_b": float(omega_b),
        "Omega_d": float(omega_m - omega_b),
        "sigma8": float(sigma8_0),
        "geo_xi": float(geo_xi),
        "mu_eff": float(mu_eff),
        "Omega_growth": float(mu_eff * omega_m),
        "chi2_fs8": float(chi2),
        "sigma8_class_raw": float(sigma8_class_raw),
        "predictions": predictions,
        "z_curve": z_curve,
        "fs8_curve": fs8_curve,
        "delta_curve": delta_curve,
        "f_curve": f_curve,
    }

    cosmo.struct_cleanup()
    cosmo.empty()

    return output


# ==================================================
# Outputs
# ==================================================

def save_outputs(lcdm, geo):
    summary_rows = []

    for result in (lcdm, geo):
        summary_rows.append({
            k: v for k, v in result.items()
            if k not in {"predictions", "z_curve", "fs8_curve", "delta_curve", "f_curve"}
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = CSV_DIR / "geo_external_growth_engine_summary.csv"
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
    pred_path = CSV_DIR / "geo_external_growth_engine_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    curve_df = pd.DataFrame({
        "z_LCDM": lcdm["z_curve"],
        "fs8_LCDM_curve": lcdm["fs8_curve"],
        "delta_LCDM": lcdm["delta_curve"],
        "f_LCDM": lcdm["f_curve"],
        "z_GEO_CLASS": geo["z_curve"],
        "fs8_GEO_CLASS_curve": geo["fs8_curve"],
        "delta_GEO_CLASS": geo["delta_curve"],
        "f_GEO_CLASS": geo["f_curve"],
    })
    curve_path = CSV_DIR / "geo_external_growth_engine_curves.csv"
    curve_df.to_csv(curve_path, index=False)

    plt.figure()
    plt.errorbar(z_data, fs8_obs, yerr=fs8_err, fmt="o", label="Observed fσ8")
    plt.plot(lcdm["z_curve"], lcdm["fs8_curve"], label="LCDM external growth")
    plt.plot(geo["z_curve"], geo["fs8_curve"], label="GEO_CLASS external growth")
    plt.xlabel("z")
    plt.ylabel("fσ8")
    plt.title("GEO external growth engine")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "geo_external_growth_engine_fs8.png", dpi=160)
    plt.close()

    plt.figure()
    plt.axhline(0.0, linestyle="--")
    plt.errorbar(z_data, pred_df["residual_LCDM"], yerr=fs8_err, fmt="o", label="LCDM residual")
    plt.errorbar(z_data, pred_df["residual_GEO_CLASS"], yerr=fs8_err, fmt="o", label="GEO_CLASS residual")
    plt.xlabel("z")
    plt.ylabel("Observed - predicted")
    plt.title("GEO external growth engine residuals")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "geo_external_growth_engine_residuals.png", dpi=160)
    plt.close()

    log_path = LOGS_DIR / "geo_external_growth_engine.log"
    log_path.write_text(
        "GEO_CLASS external growth engine completed.\n"
        "CLASS background was kept standard; GEO was applied as external growth source.\n"
        f"Delta chi2 GEO-LCDM = {geo['chi2_fs8'] - lcdm['chi2_fs8']}\n"
        f"Summary CSV: {summary_path}\n"
        f"Predictions CSV: {pred_path}\n"
        f"Curves CSV: {curve_path}\n",
        encoding="utf-8",
    )


# ==================================================
# Main
# ==================================================

def print_result(result):
    print("\nModel:", result["model"])
    for key in [
        "h",
        "Omega_m",
        "Omega_b",
        "Omega_d",
        "sigma8",
        "geo_xi",
        "mu_eff",
        "Omega_growth",
        "chi2_fs8",
        "sigma8_class_raw",
    ]:
        print(f"{key:18s} = {result[key]}")


def main():
    print("\n==============================")
    print("GEO_CLASS — EXTERNAL GROWTH ENGINE")
    print("==============================")
    print("CLASS background: standard LCDM")
    print("GEO_CLASS: external effective growth source")

    lcdm = fit_model("LCDM")
    geo = fit_model("GEO_CLASS")

    print_result(lcdm)
    print_result(geo)

    print("\n==============================")
    print("COMPARISON")
    print("==============================")
    print("Delta chi2 GEO - LCDM =", geo["chi2_fs8"] - lcdm["chi2_fs8"])

    print("\n==============================")
    print("PREDICTIONS fσ8")
    print("==============================")
    print("z      obs      err      LCDM      GEO_CLASS")

    for z, obs, err, pl, pg in zip(
        z_data,
        fs8_obs,
        fs8_err,
        lcdm["predictions"],
        geo["predictions"],
    ):
        print(f"{z:5.3f}  {obs:7.4f}  {err:7.4f}  {pl:7.4f}  {pg:7.4f}")

    save_outputs(lcdm, geo)

    print("\nSaved outputs:")
    print(" - outputs/csv/geo_external_growth_engine_summary.csv")
    print(" - outputs/csv/geo_external_growth_engine_predictions.csv")
    print(" - outputs/csv/geo_external_growth_engine_curves.csv")
    print(" - outputs/plots/geo_external_growth_engine_fs8.png")
    print(" - outputs/plots/geo_external_growth_engine_residuals.png")
    print(" - outputs/logs/geo_external_growth_engine.log\n")


if __name__ == "__main__":
    main()
