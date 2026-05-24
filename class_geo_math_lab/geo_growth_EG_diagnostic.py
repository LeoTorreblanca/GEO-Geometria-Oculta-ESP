"""
GEO_CLASS — Predictive E_G Diagnostic

Purpose:
Compare LCDM vs GEO_CLASS after calibration and extract direct predictive ratios:

- f(z)
- D(z)
- fσ8(z)
- E_G(z)
- GEO/LCDM ratios

This is not a fit.
It is a post-fit prediction diagnostic.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from classy import Class


BASE_DIR = Path(__file__).resolve().parent

CSV_DIR = BASE_DIR / "outputs" / "csv"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"
LOGS_DIR = BASE_DIR / "outputs" / "logs"

for folder in (CSV_DIR, PLOTS_DIR, LOGS_DIR):
    folder.mkdir(parents=True, exist_ok=True)


OMEGA_B = 0.05

MODELS = {
    "LCDM": {
        "h": 0.6719527376635757,
        "Omega_m": 0.31171194726388524,
        "logA": 3.033232875988148,
        "geo_xi": 1.0,
    },
    "GEO_CLASS": {
        "h": 0.6714049044158887,
        "Omega_m": 0.31424565868154797,
        "logA": 3.0440543063084116,
        "geo_xi": 0.7739826180784967,
    },
}


def run_class(params: dict):
    h = params["h"]
    omega_m = params["Omega_m"]
    geo_xi = params["geo_xi"]
    A_s = np.exp(params["logA"]) / 1e10

    omega_b = OMEGA_B * h**2
    omega_cdm = (omega_m - OMEGA_B) * h**2

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
    return cosmo


def mu_eff(omega_m: float, geo_xi: float) -> float:
    omega_d = omega_m - OMEGA_B
    return float((OMEGA_B + geo_xi * omega_d) / omega_m)


def EG(omega_m: float, fz: float, sigma: float = 1.0) -> float:
    return float(omega_m * sigma / fz)


def run_model(name: str, params: dict, z_vals: np.ndarray) -> tuple[dict, pd.DataFrame]:
    cosmo = run_class(params)

    omega_m = params["Omega_m"]
    geo_xi = params["geo_xi"]
    mu = mu_eff(omega_m, geo_xi)
    sigma8 = float(cosmo.sigma8())

    rows = []

    for z in z_vals:
        f = float(cosmo.scale_independent_growth_factor_f(float(z)))
        D = float(cosmo.scale_independent_growth_factor(float(z)))
        fs8 = float(f * sigma8 * D)
        eg = EG(omega_m, f, sigma=1.0)

        rows.append({
            "model": name,
            "z": float(z),
            "f": f,
            "D": D,
            "fs8": fs8,
            "EG_sigma1": eg,
        })

    summary = {
        "model": name,
        "Omega_m": float(omega_m),
        "Omega_b": float(OMEGA_B),
        "Omega_d": float(omega_m - OMEGA_B),
        "geo_xi": float(geo_xi),
        "mu_eff": mu,
        "sigma8": sigma8,
    }

    cosmo.struct_cleanup()
    cosmo.empty()

    return summary, pd.DataFrame(rows)


def save_outputs(summary_df: pd.DataFrame, table_df: pd.DataFrame, ratio_df: pd.DataFrame) -> None:
    summary_path = CSV_DIR / "geo_growth_EG_diagnostic_summary.csv"
    table_path = CSV_DIR / "geo_growth_EG_diagnostic_table.csv"
    ratio_path = CSV_DIR / "geo_growth_EG_diagnostic_ratios.csv"

    summary_df.to_csv(summary_path, index=False)
    table_df.to_csv(table_path, index=False)
    ratio_df.to_csv(ratio_path, index=False)

    plt.figure()
    plt.plot(ratio_df["z"], ratio_df["f_ratio"], marker="o", label="f ratio")
    plt.plot(ratio_df["z"], ratio_df["fs8_ratio"], marker="o", label="fσ8 ratio")
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("z")
    plt.ylabel("GEO_CLASS / LCDM")
    plt.title("GEO_CLASS predictive growth ratios")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "geo_growth_EG_diagnostic_growth_ratios.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(ratio_df["z"], ratio_df["EG_ratio"], marker="o", label="E_G ratio")
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("z")
    plt.ylabel("GEO_CLASS / LCDM")
    plt.title("GEO_CLASS predictive E_G ratio")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "geo_growth_EG_diagnostic_EG_ratio.png", dpi=160)
    plt.close()

    log_path = LOGS_DIR / "geo_growth_EG_diagnostic.log"
    log_path.write_text(
        "GEO_CLASS predictive E_G diagnostic completed.\n"
        f"CSV summary: {summary_path}\n"
        f"CSV table: {table_path}\n"
        f"CSV ratios: {ratio_path}\n"
        "Interpretation: fs8_ratio < 1 indicates suppressed growth; "
        "EG_ratio > 1 indicates a predicted lensing/growth deviation under Sigma=1.\n",
        encoding="utf-8",
    )


def main() -> None:
    z_vals = np.array([0.1, 0.3, 0.5, 0.7, 1.0])

    print("\n===================================")
    print("GEO_CLASS — PREDICTIVE E_G DIAGNOSTIC")
    print("===================================")

    summaries = []
    tables = {}

    for name, params in MODELS.items():
        summary, table = run_model(name, params, z_vals)
        summaries.append(summary)
        tables[name] = table

        print("\nModel:", name)
        for key, value in summary.items():
            print(f"{key:10s} = {value}")

        print("\nz      f(z)      D(z)      fs8       E_G(Sigma=1)")
        for _, row in table.iterrows():
            print(
                f"{row['z']:4.2f}   "
                f"{row['f']:8.5f}  "
                f"{row['D']:8.5f}  "
                f"{row['fs8']:8.5f}  "
                f"{row['EG_sigma1']:8.5f}"
            )

    lcdm = tables["LCDM"].set_index("z")
    geo = tables["GEO_CLASS"].set_index("z")

    ratio_df = pd.DataFrame({
        "z": z_vals,
        "f_ratio": geo["f"].values / lcdm["f"].values,
        "D_ratio": geo["D"].values / lcdm["D"].values,
        "fs8_ratio": geo["fs8"].values / lcdm["fs8"].values,
        "EG_ratio": geo["EG_sigma1"].values / lcdm["EG_sigma1"].values,
    })

    print("\n===================================")
    print("RATIOS GEO_CLASS / LCDM")
    print("===================================")
    print(ratio_df.to_string(index=False))

    print("\nInterpretation:")
    print("- If fs8_ratio < 1, GEO_CLASS predicts suppressed growth relative to LCDM.")
    print("- If EG_ratio > 1 with Sigma=1, GEO_CLASS predicts a distinct lensing/growth signal.")
    print("- If real E_G data favors LCDM, GEO_CLASS is tensioned.")
    print("- If real E_G data allows higher E_G, GEO_CLASS gains plausibility.")

    summary_df = pd.DataFrame(summaries)
    table_df = pd.concat(tables.values(), ignore_index=True)

    save_outputs(summary_df, table_df, ratio_df)

    print("\nSaved outputs:")
    print(" - outputs/csv/geo_growth_EG_diagnostic_summary.csv")
    print(" - outputs/csv/geo_growth_EG_diagnostic_table.csv")
    print(" - outputs/csv/geo_growth_EG_diagnostic_ratios.csv")
    print(" - outputs/plots/geo_growth_EG_diagnostic_growth_ratios.png")
    print(" - outputs/plots/geo_growth_EG_diagnostic_EG_ratio.png")
    print(" - outputs/logs/geo_growth_EG_diagnostic.log\n")


if __name__ == "__main__":
    main()
