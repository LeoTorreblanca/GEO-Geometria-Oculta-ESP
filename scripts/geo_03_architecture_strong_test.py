from pathlib import Path
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from geo_01_master_test import (
    TESTS,
    OMEGA_B,
    fit_model,
    run_class,
    chi2_bao,
    chi2_sn,
    chi2_fs8,
    chi2_s8,
    chi2_As,
    chi2_rd,
    chi2_ombh2,
    chi2_ommh2,
    S8_val,
    fs8_class,
)

CSV_DIR = ROOT / "resultados" / "csv"
LOG_DIR = ROOT / "resultados" / "logs"
PLOT_DIR = ROOT / "docs" / "plots" / "geo_03_architecture_strong_test"

for d in [CSV_DIR, LOG_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FC_BASE = 3.0 / 4.0
FC_NODE = math.sqrt(3.0 / 5.0)
FC_IDEAL = math.pi / 4.0

FIXED_MODELS = {
    "LCDM_geo_xi_1": 1.0,
    "GEO_base_3_4": FC_BASE,
    "GEO_node_sqrt_3_5": FC_NODE,
    "GEO_ideal_pi_4": FC_IDEAL,
}

Z_DIAG = [0.1, 0.3, 0.5, 0.7, 1.0]


def classify_delta_bic(x):
    if x < -6:
        return "strongly favored"
    if x < -2:
        return "moderately favored"
    if abs(x) <= 2:
        return "technical tie"
    if x > 6:
        return "strongly disfavored"
    return "moderately disfavored"


def fixed_geo_fit(geo_xi, test):
    from scipy.optimize import differential_evolution

    bounds = [
        (0.62, 0.75),
        (0.22, 0.42),
        (2.8, 3.3),
    ]

    def objective(x):
        h, om, logA = x

        if om <= OMEGA_B:
            return 1e30

        try:
            cosmo, _ = run_class(h, om, logA, geo_xi, "GEO_CLASS")

            total = 0.0

            if test["use_bao"]:
                total += chi2_bao(cosmo)
            if test["use_sn"]:
                total += chi2_sn(cosmo)
            if test["use_fs8"]:
                total += chi2_fs8(cosmo)
            if test["use_s8"]:
                total += chi2_s8(cosmo, om, test["s8_obs"], test["s8_err"])
            if test["use_as"]:
                total += chi2_As(logA)
            if test["use_cmb_priors"]:
                chi_rd_val, _ = chi2_rd(cosmo)
                total += chi_rd_val
                total += chi2_ombh2(h)
                total += chi2_ommh2(h, om)

            cosmo.struct_cleanup()
            cosmo.empty()
            return float(total)

        except Exception:
            return 1e30

    result = differential_evolution(
        objective,
        bounds,
        seed=123,
        polish=True,
        tol=1e-4,
        maxiter=test["maxiter"],
        popsize=test["popsize"],
    )

    h, om, logA = result.x
    cosmo, A_s = run_class(h, om, logA, geo_xi, "GEO_CLASS")

    parts = {
        "BAO": chi2_bao(cosmo) if test["use_bao"] else 0.0,
        "SN": chi2_sn(cosmo) if test["use_sn"] else 0.0,
        "fs8": chi2_fs8(cosmo) if test["use_fs8"] else 0.0,
        "S8": chi2_s8(cosmo, om, test["s8_obs"], test["s8_err"]) if test["use_s8"] else 0.0,
        "As": chi2_As(logA) if test["use_as"] else 0.0,
    }

    chi_rd_val, rd = chi2_rd(cosmo)
    parts["rd"] = chi_rd_val if test["use_cmb_priors"] else 0.0
    parts["ombh2"] = chi2_ombh2(h) if test["use_cmb_priors"] else 0.0
    parts["ommh2"] = chi2_ommh2(h, om) if test["use_cmb_priors"] else 0.0

    chi_total = sum(parts.values())

    od = om - OMEGA_B
    mu_eff = (OMEGA_B + geo_xi * od) / om

    n = 0
    if test["use_bao"]:
        n += 13
    if test["use_sn"]:
        n += 1701
    if test["use_fs8"]:
        n += 21
    if test["use_s8"]:
        n += 1
    if test["use_as"]:
        n += 1
    if test["use_cmb_priors"]:
        n += 3

    k = 3
    AIC = chi_total + 2 * k
    BIC = chi_total + k * np.log(max(n, 2))

    fs8_pred = {z: fs8_class(cosmo, z) for z in Z_DIAG}

    output = {
        "geo_xi": float(geo_xi),
        "f_out": float(1.0 - geo_xi),
        "eta": float(geo_xi ** 2),
        "strain_to_node": float(FC_NODE - geo_xi),
        "strain_to_ideal": float(FC_IDEAL - geo_xi),
        "H0": float(100 * h),
        "h": float(h),
        "Omega_m": float(om),
        "Omega_b": float(OMEGA_B),
        "Omega_d": float(od),
        "logA": float(logA),
        "A_s": float(A_s),
        "sigma8": float(cosmo.sigma8()),
        "S8": float(S8_val(cosmo, om)),
        "rd": float(rd),
        "mu_eff": float(mu_eff),
        "Omega_growth": float(mu_eff * om),
        "chi_total": float(chi_total),
        "AIC": float(AIC),
        "BIC": float(BIC),
        **{f"chi_{k2}": float(v) for k2, v in parts.items()},
        **{f"fs8_z_{z}": float(v) for z, v in fs8_pred.items()},
    }

    cosmo.struct_cleanup()
    cosmo.empty()

    return output


def save_outputs(rows, summary_rows):
    df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summary_rows)

    detail_path = CSV_DIR / "geo_03_architecture_strong_test_details.csv"
    summary_path = CSV_DIR / "geo_03_architecture_strong_test_summary.csv"

    df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    tests = summary_df["test"].tolist()

    plt.figure(figsize=(11, 6))
    plt.bar(tests, summary_df["delta_node_vs_free_geo"])
    plt.axhline(0)
    plt.axhline(2, linestyle="--")
    plt.axhline(-2, linestyle="--")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Delta BIC node - free GEO")
    plt.title("GEO 03 — Fixed sqrt(3/5) node vs free GEO")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "01_delta_bic_node_vs_free_geo.png", dpi=200)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.bar(tests, summary_df["delta_base_vs_free_geo"])
    plt.axhline(0)
    plt.axhline(2, linestyle="--")
    plt.axhline(-2, linestyle="--")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Delta BIC base - free GEO")
    plt.title("GEO 03 — Fixed 3/4 base vs free GEO")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "02_delta_bic_base_vs_free_geo.png", dpi=200)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.plot(tests, summary_df["node_S8"], "o-", label="sqrt(3/5) node S8")
    plt.plot(tests, summary_df["free_geo_S8"], "o-", label="free GEO S8")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("S8")
    plt.title("GEO 03 — S8 fixed node vs free GEO")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "03_s8_node_vs_free_geo.png", dpi=200)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.plot(tests, summary_df["node_mu_eff"], "o-", label="node mu_eff")
    plt.plot(tests, summary_df["free_geo_mu_eff"], "o-", label="free GEO mu_eff")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("mu_eff")
    plt.title("GEO 03 — Effective growth coupling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "04_mu_eff_node_vs_free_geo.png", dpi=200)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.bar(tests, summary_df["best_fixed_delta_lcdm"])
    plt.axhline(0)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Delta BIC best fixed - fixed LCDM")
    plt.title("GEO 03 — Best fixed geometry vs fixed LCDM")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "05_best_fixed_vs_lcdm.png", dpi=200)
    plt.close()

    log_path = LOG_DIR / "geo_03_architecture_strong_test.log"
    log_path.write_text(
        "GEO 03 Architecture Strong Test completed.\n"
        f"Details CSV: {detail_path}\n"
        f"Summary CSV: {summary_path}\n"
        f"Plots: {PLOT_DIR}\n",
        encoding="utf-8",
    )


def main():
    print("\n================================================")
    print("GEO 03 — ARCHITECTURE STRONG TEST")
    print("Fixed geometric couplings vs free GEO_CLASS")
    print("================================================")

    all_rows = []
    summary_rows = []

    for test in TESTS:
        print("\n\n################################################")
        print("TEST:", test["name"])
        print("################################################")

        lcdm_free = fit_model("LCDM", test)
        geo_free = fit_model("GEO_CLASS", test)
        mu_free = fit_model("FREE_MU", test)

        fixed = {}

        for name, geo_xi in FIXED_MODELS.items():
            print("Running fixed:", name, "geo_xi=", geo_xi)
            fixed[name] = fixed_geo_fit(geo_xi, test)

        best_fixed_name, best_fixed = min(fixed.items(), key=lambda kv: kv[1]["BIC"])

        print("\nFREE REFERENCES")
        print("LCDM free BIC      =", lcdm_free["BIC"])
        print("GEO free BIC       =", geo_free["BIC"])
        print("GEO free geo_xi    =", geo_free["geo_xi"])
        print("GEO free eta       =", geo_free["geo_xi"] ** 2)
        print("FREE_MU BIC        =", mu_free["BIC"])
        print("FREE_MU xi_equiv   =", mu_free["xi_equiv"])

        print("\nFIXED GEOMETRIC MODELS")
        for name, result in fixed.items():
            delta_lcdm = result["BIC"] - fixed["LCDM_geo_xi_1"]["BIC"]
            delta_free = result["BIC"] - geo_free["BIC"]

            print("\nModel:", name)
            print("geo_xi              =", result["geo_xi"])
            print("eta                 =", result["eta"])
            print("H0                  =", result["H0"])
            print("Omega_m             =", result["Omega_m"])
            print("S8                  =", result["S8"])
            print("mu_eff              =", result["mu_eff"])
            print("Omega_growth        =", result["Omega_growth"])
            print("chi_total           =", result["chi_total"])
            print("BIC                 =", result["BIC"])
            print("Delta BIC vs LCDM   =", delta_lcdm, classify_delta_bic(delta_lcdm))
            print("Delta BIC vs free GEO =", delta_free, classify_delta_bic(delta_free))

            all_rows.append({
                "test": test["name"],
                "fixed_model": name,
                **result,
                "free_geo_BIC": geo_free["BIC"],
                "free_geo_xi": geo_free["geo_xi"],
                "delta_vs_fixed_lcdm": delta_lcdm,
                "delta_vs_free_geo": delta_free,
            })

        node = fixed["GEO_node_sqrt_3_5"]
        base = fixed["GEO_base_3_4"]
        ideal = fixed["GEO_ideal_pi_4"]

        summary = {
            "test": test["name"],
            "free_geo_xi": geo_free["geo_xi"],
            "free_geo_eta": geo_free["geo_xi"] ** 2,
            "free_geo_BIC": geo_free["BIC"],
            "free_geo_S8": geo_free["S8"],
            "free_geo_mu_eff": geo_free["mu_eff"],
            "best_fixed": best_fixed_name,
            "best_fixed_BIC": best_fixed["BIC"],
            "best_fixed_delta_lcdm": best_fixed["BIC"] - fixed["LCDM_geo_xi_1"]["BIC"],
            "delta_node_vs_free_geo": node["BIC"] - geo_free["BIC"],
            "delta_base_vs_free_geo": base["BIC"] - geo_free["BIC"],
            "delta_ideal_vs_free_geo": ideal["BIC"] - geo_free["BIC"],
            "node_S8": node["S8"],
            "node_mu_eff": node["mu_eff"],
            "node_Omega_growth": node["Omega_growth"],
            "base_S8": base["S8"],
            "base_mu_eff": base["mu_eff"],
            "ideal_S8": ideal["S8"],
            "ideal_mu_eff": ideal["mu_eff"],
        }

        summary_rows.append(summary)

        print("\nARCHITECTURE DIAGNOSTIC")
        print("Best fixed =", best_fixed_name)
        print("Delta BIC best fixed vs fixed LCDM =", summary["best_fixed_delta_lcdm"])
        print("Delta BIC node vs free GEO =", summary["delta_node_vs_free_geo"])
        print("Delta BIC base vs free GEO =", summary["delta_base_vs_free_geo"])
        print("Delta BIC ideal vs free GEO =", summary["delta_ideal_vs_free_geo"])

        if abs(summary["delta_node_vs_free_geo"]) <= 2:
            print("NODE SURVIVES: sqrt(3/5) can replace free GEO within technical tie.")
        else:
            print("NODE TENSIONED: sqrt(3/5) loses against free GEO.")

        if abs(summary["delta_base_vs_free_geo"]) <= 2:
            print("BASE SURVIVES: 3/4 can replace free GEO within technical tie.")
        else:
            print("BASE TENSIONED: 3/4 loses against free GEO.")

    save_outputs(all_rows, summary_rows)

    print("\nOutputs:")
    print(" - resultados/csv/geo_03_architecture_strong_test_details.csv")
    print(" - resultados/csv/geo_03_architecture_strong_test_summary.csv")
    print(" - docs/plots/geo_03_architecture_strong_test/")
    print(" - resultados/logs/geo_03_architecture_strong_test.log")


if __name__ == "__main__":
    main()
